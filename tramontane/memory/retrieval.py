"""4-channel memory retrieval with Reciprocal Rank Fusion.

Channels:
1. Semantic: cosine similarity on mistral-embed vectors
2. Keyword: FTS5 BM25 full-text search
3. Entity: graph traversal from mentioned entities
4. Temporal: recency + access frequency + confidence decay
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

RRF_K = 60  # Reciprocal Rank Fusion constant


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _extract_entities(text: str) -> list[str]:
    """Simple entity extraction: capitalized words and quoted phrases."""
    entities: list[str] = []
    # Quoted phrases
    entities.extend(re.findall(r'"([^"]+)"', text))
    entities.extend(re.findall(r"'([^']+)'", text))
    # Capitalized words (likely proper nouns), skip sentence starts
    words = text.split()
    for i, word in enumerate(words):
        clean = word.strip(".,;:!?()[]")
        if clean and clean[0].isupper() and i > 0 and len(clean) > 1:
            entities.append(clean)
    return list(set(entities))


class MemoryRetriever:
    """4-channel retrieval with RRF fusion."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_model: str = "mistral-embed",
    ) -> None:
        self._conn = conn
        self._embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        recency_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve memories using 4-channel fusion."""
        if not query.strip():
            return []

        # Channel 1: Semantic search
        semantic_ids = await self._semantic_search(query, top_k * 2)

        # Channel 2: Keyword search (FTS5)
        keyword_ids = self._keyword_search(query, top_k * 2)

        # Channel 3: Entity search
        entity_ids = self._entity_search(query, top_k * 2)

        # Channel 4: Temporal scoring (applied as a boost, not a channel)
        channels = [semantic_ids, keyword_ids, entity_ids]

        # RRF fusion
        fused = self._fuse_results(channels)

        # Apply temporal boost
        for mem_id in fused:
            row = self._conn.execute(
                "SELECT updated_at, access_count, confidence FROM factual_memory WHERE id=?",
                (mem_id,),
            ).fetchone()
            if row:
                temporal = self._temporal_score(row)
                fused[mem_id] = (
                    fused[mem_id] * (1 - recency_weight)
                    + temporal * recency_weight
                )

        # Sort and return top_k
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for mem_id, score in ranked:
            row = self._conn.execute(
                "SELECT id, content, entity, category, source, created_at "
                "FROM factual_memory WHERE id=? AND erased_at IS NULL",
                (mem_id,),
            ).fetchone()
            if row:
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "entity": row["entity"] or "",
                    "category": row["category"] or "fact",
                    "score": round(score, 4),
                    "source": row["source"] or "",
                    "created_at": row["created_at"] or "",
                })
        return results

    async def _semantic_search(
        self, query: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Embed query and find most similar memories."""
        from mistralai.client import Mistral

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))
        resp = await client.embeddings.create_async(
            model=self._embedding_model, inputs=[query],
        )
        query_emb = resp.data[0].embedding if resp.data[0].embedding else []
        if not query_emb:
            return []

        rows = self._conn.execute(
            "SELECT id, embedding FROM factual_memory "
            "WHERE erased_at IS NULL AND embedding IS NOT NULL",
        ).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            stored = json.loads(row["embedding"]) if row["embedding"] else []
            if stored:
                sim = _cosine_similarity(query_emb, stored)
                scored.append((row["id"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _keyword_search(
        self, query: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 keyword search."""
        try:
            # FTS5 MATCH with BM25 ranking
            safe_query = " OR ".join(
                word for word in query.split() if word.isalnum()
            )
            if not safe_query:
                return []
            rows = self._conn.execute(
                "SELECT id, rank FROM factual_memory_fts WHERE factual_memory_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (safe_query, top_k),
            ).fetchall()
            # FTS5 rank is negative (lower = better), normalize
            if not rows:
                return []
            results: list[tuple[str, float]] = []
            for row in rows:
                # Convert negative rank to positive score
                score = 1.0 / (1.0 + abs(float(row["rank"])))
                results.append((row["id"], score))
            return results
        except sqlite3.OperationalError:
            logger.debug("FTS5 search failed for query: %s", query[:50])
            return []

    def _entity_search(
        self, query: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Search via entity graph links."""
        entities = _extract_entities(query)
        if not entities:
            return []

        results: list[tuple[str, float]] = []
        seen: set[str] = set()

        for entity in entities:
            # Direct match: facts with this entity
            rows = self._conn.execute(
                "SELECT id FROM factual_memory WHERE entity = ? AND erased_at IS NULL",
                (entity,),
            ).fetchall()
            for row in rows:
                if row["id"] not in seen:
                    results.append((row["id"], 1.0))
                    seen.add(row["id"])

            # 1-hop: follow entity links
            links = self._conn.execute(
                "SELECT target_entity, weight FROM entity_links WHERE source_entity = ? "
                "UNION SELECT source_entity, weight FROM entity_links WHERE target_entity = ?",
                (entity, entity),
            ).fetchall()
            for link in links:
                linked_rows = self._conn.execute(
                    "SELECT id FROM factual_memory WHERE entity = ? AND erased_at IS NULL",
                    (link["target_entity"],),
                ).fetchall()
                for row in linked_rows:
                    if row["id"] not in seen:
                        results.append((row["id"], float(link["weight"]) * 0.5))
                        seen.add(row["id"])

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _temporal_score(row: sqlite3.Row) -> float:
        """Score based on recency, access frequency, confidence."""
        updated = row["updated_at"] or ""
        access_count = int(row["access_count"] or 0)
        confidence = float(row["confidence"] or 1.0)

        # Recency: days since update
        days = 1.0
        if updated:
            try:
                import datetime

                updated_dt = datetime.datetime.fromisoformat(updated)
                days = max(1.0, (datetime.datetime.now() - updated_dt).days + 1)
            except (ValueError, TypeError):
                pass

        recency = 1.0 / days
        frequency = min(math.log(access_count + 1) + 0.5, 2.0)
        return recency * frequency * confidence

    @staticmethod
    def _fuse_results(
        channels: list[list[tuple[str, float]]],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion across channels."""
        scores: dict[str, float] = {}
        for channel in channels:
            for rank, (mem_id, _channel_score) in enumerate(channel):
                scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return scores
