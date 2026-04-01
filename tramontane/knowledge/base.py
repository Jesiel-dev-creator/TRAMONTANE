"""Knowledge base — RAG with mistral-embed for grounded agents."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB = "tramontane_knowledge.db"
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 1000
DEFAULT_TOP_K = 5


@dataclass
class Chunk:
    """A text chunk with its embedding."""

    id: str
    content: str
    source: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from a knowledge base query."""

    chunks: list[Chunk]
    query: str
    scores: list[float]


class KnowledgeBase:
    """RAG knowledge base using mistral-embed for embeddings.

    Stores chunks + embeddings in SQLite. No external vector DB needed.
    Uses cosine similarity for retrieval.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        embedding_model: str = "mistral-embed",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._db_path = db_path
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_kb_source
                ON kb_chunks(source);
        """)
        self._conn.commit()

    def _chunk_text(self, text: str, source: str) -> list[Chunk]:
        """Split text into overlapping chunks."""
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)
                if break_point > self._chunk_size * 0.5:
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + break_point + 1

            chunk_id = hashlib.sha256(
                f"{source}:{start}:{chunk_text[:100]}".encode(),
            ).hexdigest()[:16]

            chunks.append(
                Chunk(id=chunk_id, content=chunk_text.strip(), source=source),
            )

            start = end - self._chunk_overlap
            if start <= 0 and end >= len(text):
                break

        return chunks

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Mistral API."""
        from mistralai.client import Mistral

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        all_embeddings: list[list[float]] = []
        batch_size = 128

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await client.embeddings.create_async(
                model=self._embedding_model,
                inputs=batch,
            )
            all_embeddings.extend([
                d.embedding for d in response.data if d.embedding is not None
            ])

        return all_embeddings

    async def ingest(
        self,
        sources: list[str] | None = None,
        texts: list[tuple[str, str]] | None = None,
    ) -> int:
        """Ingest documents into the knowledge base.

        Args:
            sources: File paths or glob patterns.
            texts: List of (content, source_name) tuples.

        Returns:
            Number of chunks ingested.
        """
        import glob as glob_mod

        all_chunks: list[Chunk] = []

        if sources:
            allowed = {".md", ".txt", ".py", ".html", ".json", ".yaml", ".yml"}
            for pattern in sources:
                for filepath in glob_mod.glob(pattern, recursive=True):
                    path = Path(filepath)
                    if path.is_file() and path.suffix in allowed:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                        chunks = self._chunk_text(text, str(path))
                        all_chunks.extend(chunks)
                        logger.info("Chunked %s: %d chunks", path, len(chunks))

        if texts:
            for content, source_name in texts:
                chunks = self._chunk_text(content, source_name)
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks to ingest")
            return 0

        logger.info(
            "Embedding %d chunks with %s...",
            len(all_chunks),
            self._embedding_model,
        )
        embeddings = await self._embed_batch([c.content for c in all_chunks])

        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
            self._conn.execute(
                """INSERT OR REPLACE INTO kb_chunks
                   (id, content, source, embedding, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    chunk.id,
                    chunk.content,
                    chunk.source,
                    json.dumps(embedding),
                    json.dumps(chunk.metadata),
                ),
            )
        self._conn.commit()

        logger.info("Ingested %d chunks into %s", len(all_chunks), self._db_path)
        return len(all_chunks)

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query."""
        query_embedding = (await self._embed_batch([query]))[0]

        rows = self._conn.execute(
            "SELECT id, content, source, embedding, metadata FROM kb_chunks",
        ).fetchall()

        scored: list[tuple[float, Chunk]] = []
        for row in rows:
            stored_embedding: list[float] = json.loads(row["embedding"])
            score = self._cosine_similarity(query_embedding, stored_embedding)
            chunk = Chunk(
                id=row["id"],
                content=row["content"],
                source=row["source"],
                embedding=stored_embedding,
                metadata=json.loads(row["metadata"]),
            )
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return RetrievalResult(
            query=query,
            chunks=[chunk for _, chunk in top],
            scores=[score for score, _ in top],
        )

    @staticmethod
    def format_context(result: RetrievalResult) -> str:
        """Format retrieved chunks as context for prompt injection."""
        if not result.chunks:
            return ""

        parts = ["## Retrieved Context (from knowledge base)\n"]
        for chunk, score in zip(result.chunks, result.scores):
            parts.append(
                f"### Source: {chunk.source} (relevance: {score:.2f})\n"
                f"{chunk.content}\n",
            )
        return "\n".join(parts)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the knowledge base."""
        row = self._conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()
        return row[0] if row else 0
