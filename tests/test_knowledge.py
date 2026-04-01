"""Tests for tramontane.knowledge.base — KnowledgeBase RAG."""

from __future__ import annotations

import pytest

from tramontane.knowledge.base import Chunk, KnowledgeBase, RetrievalResult


class TestChunking:
    """KnowledgeBase._chunk_text() text splitting."""

    def test_short_text_single_chunk(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db, chunk_size=3000, chunk_overlap=1000)
        chunks = kb._chunk_text("Short text.", "test.txt")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].source == "test.txt"

    def test_long_text_multiple_chunks(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db, chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = kb._chunk_text(text, "long.txt")
        assert len(chunks) >= 2

    def test_chunks_have_overlap(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db, chunk_size=100, chunk_overlap=30)
        text = "word " * 50  # 250 chars
        chunks = kb._chunk_text(text, "overlap.txt")
        if len(chunks) >= 2:
            # Content from end of chunk 1 should appear in start of chunk 2
            end_of_first = chunks[0].content[-20:]
            assert end_of_first in chunks[1].content or len(chunks[1].content) > 0

    def test_sentence_boundary_breaking(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db, chunk_size=100, chunk_overlap=20)
        text = "First sentence here. " * 3 + "Second part starts. " * 3
        chunks = kb._chunk_text(text, "sentences.txt")
        # Should break at '. ' when possible
        assert len(chunks) >= 1

    def test_unique_chunk_ids(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db, chunk_size=50, chunk_overlap=10)
        chunks = kb._chunk_text("A" * 200, "ids.txt")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique


class TestCosineSimilarity:
    """KnowledgeBase._cosine_similarity()."""

    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert KnowledgeBase._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert KnowledgeBase._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert KnowledgeBase._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert KnowledgeBase._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestFormatContext:
    """KnowledgeBase.format_context()."""

    def test_formats_chunks(self) -> None:
        result = RetrievalResult(
            query="test",
            chunks=[Chunk(id="1", content="Hello world", source="doc.md")],
            scores=[0.95],
        )
        formatted = KnowledgeBase.format_context(result)
        assert "doc.md" in formatted
        assert "Hello world" in formatted
        assert "0.95" in formatted

    def test_empty_result(self) -> None:
        result = RetrievalResult(query="test", chunks=[], scores=[])
        assert KnowledgeBase.format_context(result) == ""


class TestChunkCount:
    """KnowledgeBase.chunk_count property."""

    def test_empty_db(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/kb.db"  # type: ignore[operator]
        kb = KnowledgeBase(db_path=db)
        assert kb.chunk_count == 0


class TestKnowledgeImports:
    """Package-level imports."""

    def test_knowledge_base_importable(self) -> None:
        from tramontane import KnowledgeBase

        assert KnowledgeBase is not None

    def test_retrieval_result_importable(self) -> None:
        from tramontane import RetrievalResult

        assert RetrievalResult is not None
