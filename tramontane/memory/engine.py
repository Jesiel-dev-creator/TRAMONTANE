"""TramontaneMemory — 3-tier agent-controlled memory system.

Tier 1: Working Memory (always in context, agent-editable)
Tier 2: Factual Memory (extracted facts, vector + FTS5 + entity graph)
Tier 3: Experiential Memory (outcomes, learnings, self-improvement)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB = "tramontane_memory_v2.db"
DEDUP_THRESHOLD = 0.92  # cosine similarity above this = duplicate


@dataclass
class MemoryStats:
    """Statistics about the memory system."""

    fact_count: int = 0
    experience_count: int = 0
    working_block_count: int = 0
    entity_link_count: int = 0


class TramontaneMemory:
    """3-tier agent-controlled memory with Mistral embeddings.

    Usage:
        memory = TramontaneMemory(db_path="memory.db")
        agent = Agent(role="Gerald", memory=memory, memory_tools=True)
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        embedding_model: str = "mistral-embed",
        extraction_model: str = "ministral-3b-latest",
    ) -> None:
        self._db_path = db_path
        self._embedding_model = embedding_model
        self._extraction_model = extraction_model
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create all tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                label TEXT NOT NULL,
                content TEXT NOT NULL,
                max_tokens INTEGER DEFAULT 500,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS factual_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                entity TEXT,
                category TEXT DEFAULT 'fact',
                embedding BLOB,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                erased_at DATETIME
            );
            CREATE TABLE IF NOT EXISTS entity_links (
                id TEXT PRIMARY KEY,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relationship TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS factual_memory_fts
                USING fts5(id, content, entity, category);
            CREATE TABLE IF NOT EXISTS experiential_memory (
                id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                action_summary TEXT NOT NULL,
                outcome TEXT,
                outcome_score REAL,
                insight TEXT,
                insight_embedding BLOB,
                context_tags TEXT,
                agent_role TEXT,
                model_used TEXT,
                cost_eur REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_erasure_log (
                id TEXT PRIMARY KEY,
                memory_ids TEXT NOT NULL,
                reason TEXT,
                erased_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_audit (
                id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                memory_type TEXT,
                memory_id TEXT,
                agent_role TEXT,
                details TEXT,
                cost_eur REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    # ── Embedding helpers ────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Mistral API."""
        import os

        from mistralai.client import Mistral

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        result: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            resp = await client.embeddings.create_async(
                model=self._embedding_model, inputs=batch,
            )
            result.extend([
                d.embedding for d in resp.data if d.embedding is not None
            ])
        return result

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # ── Tier 2: Factual Memory ───────────────────────────────────────

    async def retain(
        self,
        content: str,
        entity: str = "",
        category: str = "fact",
        source: str = "",
    ) -> str:
        """Store a fact. Deduplicates by embedding similarity."""
        embedding = (await self._embed([content]))[0]

        # Dedup check: find similar existing facts
        rows = self._conn.execute(
            "SELECT id, embedding FROM factual_memory WHERE erased_at IS NULL",
        ).fetchall()
        for row in rows:
            stored = json.loads(row["embedding"]) if row["embedding"] else []
            if stored and self._cosine_sim(embedding, stored) > DEDUP_THRESHOLD:
                # Update existing instead of creating duplicate
                self._conn.execute(
                    "UPDATE factual_memory SET content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (content, row["id"]),
                )
                self._conn.execute(
                    "UPDATE factual_memory_fts SET content=? WHERE id=?",
                    (content, row["id"]),
                )
                self._conn.commit()
                logger.info("Deduplicated memory %s", row["id"])
                return str(row["id"])

        mem_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """INSERT INTO factual_memory
               (id, content, entity, category, embedding, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (mem_id, content, entity, category, json.dumps(embedding), source),
        )
        self._conn.execute(
            "INSERT INTO factual_memory_fts (id, content, entity, category) VALUES (?, ?, ?, ?)",
            (mem_id, content, entity, category),
        )
        self._conn.commit()
        self._audit("retain", "factual", mem_id, content[:100])
        return mem_id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str = "all",
        recency_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories using 4-channel fusion."""
        from tramontane.memory.retrieval import MemoryRetriever

        retriever = MemoryRetriever(self._conn, self._embedding_model)
        results = await retriever.retrieve(query, top_k, recency_weight)

        # Bump access counts
        for r in results:
            self._conn.execute(
                "UPDATE factual_memory SET access_count = access_count + 1 WHERE id = ?",
                (r["id"],),
            )
        self._conn.commit()
        return results

    async def reflect(self, question: str) -> str:
        """Synthesize insights from memories to answer a question."""
        import os

        from mistralai.client import Mistral

        related = await self.recall(question, top_k=10)
        if not related:
            return "No memories to reflect on."

        context = "\n".join(
            f"- [{m['category']}] {m['content']}" for m in related
        )

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        resp = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[  # type: ignore[arg-type]
                {
                    "role": "system",
                    "content": (
                        "You are a reflection engine. Synthesize the memories "
                        "below to answer the question. Be concise and insightful."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Memories:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
        )
        return str(resp.choices[0].message.content or "")

    async def forget(self, memory_id: str, reason: str = "") -> bool:
        """Soft-delete a memory (GDPR Article 17)."""
        row = self._conn.execute(
            "SELECT id FROM factual_memory WHERE id = ?", (memory_id,),
        ).fetchone()
        if not row:
            return False

        self._conn.execute(
            "UPDATE factual_memory SET erased_at = CURRENT_TIMESTAMP WHERE id = ?",
            (memory_id,),
        )
        self._conn.execute(
            "DELETE FROM factual_memory_fts WHERE id = ?", (memory_id,),
        )
        self._conn.execute(
            "INSERT INTO memory_erasure_log (id, memory_ids, reason) VALUES (?, ?, ?)",
            (uuid.uuid4().hex[:12], memory_id, reason),
        )
        self._conn.commit()
        self._audit("forget", "factual", memory_id, reason)
        return True

    async def update(self, memory_id: str, new_content: str) -> bool:
        """Update a memory's content and re-embed."""
        row = self._conn.execute(
            "SELECT id FROM factual_memory WHERE id = ? AND erased_at IS NULL",
            (memory_id,),
        ).fetchone()
        if not row:
            return False

        embedding = (await self._embed([new_content]))[0]
        self._conn.execute(
            """UPDATE factual_memory
               SET content=?, embedding=?, updated_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (new_content, json.dumps(embedding), memory_id),
        )
        self._conn.execute(
            "UPDATE factual_memory_fts SET content=? WHERE id=?",
            (new_content, memory_id),
        )
        self._conn.commit()
        self._audit("update", "factual", memory_id, new_content[:100])
        return True

    # ── Fact extraction ──────────────────────────────────────────────

    async def extract_facts(
        self, text: str, source: str = "",
    ) -> list[str]:
        """Auto-extract facts from text and store them."""
        from tramontane.memory.extraction import FactExtractor

        extractor = FactExtractor(model=self._extraction_model)
        facts = await extractor.extract(text)

        ids: list[str] = []
        for fact in facts:
            mem_id = await self.retain(
                fact.content, entity=fact.entity,
                category=fact.category, source=source,
            )
            ids.append(mem_id)

        logger.info("Extracted %d facts from %s", len(ids), source or "text")
        return ids

    # ── Tier 3: Experiential Memory ──────────────────────────────────

    async def record_experience(
        self,
        action_type: str,
        summary: str,
        outcome: str = "",
        score: float = 0.0,
        agent_role: str = "",
        model: str = "",
        cost: float = 0.0,
    ) -> str:
        """Record an experience (outcome of an action)."""
        exp_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """INSERT INTO experiential_memory
               (id, action_type, action_summary, outcome, outcome_score,
                agent_role, model_used, cost_eur)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, action_type, summary, outcome, score,
             agent_role, model, cost),
        )
        self._conn.commit()
        self._audit("record", "experiential", exp_id, summary[:100])
        return exp_id

    # ── Tier 1: Working Memory ───────────────────────────────────────

    def get_working_blocks(self, agent_id: str) -> list[dict[str, Any]]:
        """Get all working memory blocks for an agent."""
        from tramontane.memory.working import WorkingMemoryManager

        mgr = WorkingMemoryManager(self._conn)
        blocks = mgr.get_blocks(agent_id)
        return [
            {"label": b.label, "content": b.content, "id": b.id}
            for b in blocks
        ]

    def set_working_block(
        self, agent_id: str, label: str, content: str,
    ) -> None:
        """Set a working memory block."""
        from tramontane.memory.working import WorkingMemoryManager

        mgr = WorkingMemoryManager(self._conn)
        mgr.set_block(agent_id, label, content)

    # ── Context formatting ───────────────────────────────────────────

    @staticmethod
    def format_context(
        results: list[dict[str, Any]], max_tokens: int = 2000,
    ) -> str:
        """Format recalled memories for prompt injection."""
        if not results:
            return ""
        parts = ["## Recalled Memories\n"]
        total_chars = 0
        char_limit = max_tokens * 4  # ~4 chars per token
        for m in results:
            entry = (
                f"- [{m.get('category', 'fact')}] "
                f"{m.get('content', '')} "
                f"(score: {m.get('score', 0):.2f})\n"
            )
            if total_chars + len(entry) > char_limit:
                break
            parts.append(entry)
            total_chars += len(entry)
        return "".join(parts)

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> MemoryStats:
        """Get memory system statistics."""
        fact_count = self._conn.execute(
            "SELECT COUNT(*) FROM factual_memory WHERE erased_at IS NULL",
        ).fetchone()[0]
        exp_count = self._conn.execute(
            "SELECT COUNT(*) FROM experiential_memory",
        ).fetchone()[0]
        wm_count = self._conn.execute(
            "SELECT COUNT(*) FROM working_memory",
        ).fetchone()[0]
        link_count = self._conn.execute(
            "SELECT COUNT(*) FROM entity_links",
        ).fetchone()[0]
        return MemoryStats(
            fact_count=fact_count,
            experience_count=exp_count,
            working_block_count=wm_count,
            entity_link_count=link_count,
        )

    @property
    def fact_count(self) -> int:
        """Number of active facts."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM factual_memory WHERE erased_at IS NULL",
        ).fetchone()
        return int(row[0]) if row else 0

    @property
    def experience_count(self) -> int:
        """Number of experiences."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM experiential_memory",
        ).fetchone()
        return int(row[0]) if row else 0

    # ── Audit ────────────────────────────────────────────────────────

    def _audit(
        self, op: str, mem_type: str, mem_id: str, details: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO memory_audit
               (id, operation, memory_type, memory_id, details)
               VALUES (?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex[:12], op, mem_type, mem_id, details),
        )
        self._conn.commit()
