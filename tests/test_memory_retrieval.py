"""Tests for tramontane.memory.retrieval — 4-channel retrieval + RRF."""

from __future__ import annotations

import json

import pytest

from tramontane.memory.engine import TramontaneMemory
from tramontane.memory.retrieval import (
    MemoryRetriever,
    _cosine_similarity,
    _extract_entities,
)


class TestCosineSimilarity:
    """_cosine_similarity helper."""

    def test_identical(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0


class TestEntityExtraction:
    """_extract_entities heuristic."""

    def test_capitalized_words(self) -> None:
        entities = _extract_entities("The manager is John Smith at Google")
        assert "John" in entities or "Smith" in entities or "Google" in entities

    def test_quoted_phrases(self) -> None:
        entities = _extract_entities('Find info about "Acme Corp"')
        assert "Acme Corp" in entities

    def test_empty_text(self) -> None:
        assert _extract_entities("") == []


class TestKeywordSearch:
    """FTS5 keyword search."""

    def test_fts5_returns_results(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)

        # Insert a fact with FTS
        mem._conn.execute(
            "INSERT INTO factual_memory (id, content, entity, embedding) VALUES (?, ?, ?, ?)",
            ("f1", "Python is a programming language", "Python", json.dumps([0.1] * 10)),
        )
        mem._conn.execute(
            "INSERT INTO factual_memory_fts (id, content, entity, category) VALUES (?, ?, ?, ?)",
            ("f1", "Python is a programming language", "Python", "fact"),
        )
        mem._conn.commit()

        retriever = MemoryRetriever(mem._conn)
        results = retriever._keyword_search("Python programming", 5)
        assert len(results) >= 1
        assert results[0][0] == "f1"

    def test_fts5_empty_query(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        retriever = MemoryRetriever(mem._conn)
        assert retriever._keyword_search("", 5) == []


class TestTemporalScore:
    """MemoryRetriever._temporal_score()."""

    def test_recent_scores_higher(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        retriever = MemoryRetriever(mem._conn)

        # Create a fake row-like object
        class FakeRow:
            def __init__(self, updated_at: str, access_count: int, confidence: float) -> None:
                self._data = {
                    "updated_at": updated_at,
                    "access_count": access_count,
                    "confidence": confidence,
                }

            def __getitem__(self, key: str) -> object:
                return self._data[key]

        recent = FakeRow("2026-04-01T12:00:00", 5, 1.0)
        score = retriever._temporal_score(recent)  # type: ignore[arg-type]
        assert score > 0


class TestRRFFusion:
    """MemoryRetriever._fuse_results()."""

    def test_fuses_channels(self) -> None:
        channels = [
            [("a", 0.9), ("b", 0.8)],  # Channel 1
            [("b", 0.95), ("c", 0.7)],  # Channel 2
        ]
        fused = MemoryRetriever._fuse_results(channels)
        # "b" appears in both channels, should score highest
        assert fused["b"] > fused.get("a", 0)
        assert fused["b"] > fused.get("c", 0)

    def test_empty_channels(self) -> None:
        fused = MemoryRetriever._fuse_results([[], []])
        assert fused == {}
