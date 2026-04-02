"""Tests for tramontane.memory.engine — TramontaneMemory."""

from __future__ import annotations

import pytest

from tramontane.memory.engine import TramontaneMemory


class TestInit:
    """TramontaneMemory initialization."""

    def test_creates_tables(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        # Verify tables exist
        tables = mem._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t["name"] for t in tables}
        assert "factual_memory" in names
        assert "working_memory" in names
        assert "experiential_memory" in names
        assert "entity_links" in names
        assert "memory_erasure_log" in names
        assert "memory_audit" in names

    def test_fact_count_empty(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        assert mem.fact_count == 0

    def test_experience_count_empty(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        assert mem.experience_count == 0


class TestRetain:
    """TramontaneMemory.retain() — stores facts."""

    def test_retain_stores_directly(self, tmp_path: object) -> None:
        """Retain without API (insert directly to test storage)."""
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        # Direct insert (bypassing embedding for unit test)
        import json
        import uuid

        mem_id = uuid.uuid4().hex[:12]
        mem._conn.execute(
            "INSERT INTO factual_memory (id, content, entity, category, embedding, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mem_id, "Paris is the capital of France", "France", "fact",
             json.dumps([0.1] * 10), "test"),
        )
        mem._conn.execute(
            "INSERT INTO factual_memory_fts (id, content, entity, category) VALUES (?, ?, ?, ?)",
            (mem_id, "Paris is the capital of France", "France", "fact"),
        )
        mem._conn.commit()
        assert mem.fact_count == 1


class TestForget:
    """TramontaneMemory.forget() — soft-delete with GDPR."""

    @pytest.mark.asyncio
    async def test_forget_soft_deletes(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        import json
        import uuid

        mem_id = uuid.uuid4().hex[:12]
        mem._conn.execute(
            "INSERT INTO factual_memory (id, content, entity, embedding) VALUES (?, ?, ?, ?)",
            (mem_id, "test fact", "test", json.dumps([0.1] * 10)),
        )
        mem._conn.commit()
        assert mem.fact_count == 1

        result = await mem.forget(mem_id, reason="GDPR request")
        assert result is True
        assert mem.fact_count == 0

        log = mem._conn.execute("SELECT * FROM memory_erasure_log").fetchone()
        assert log is not None
        assert mem_id in log["memory_ids"]

    @pytest.mark.asyncio
    async def test_forget_nonexistent_returns_false(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        result = await mem.forget("nonexistent")
        assert result is False


class TestRecordExperience:
    """TramontaneMemory.record_experience()."""

    @pytest.mark.asyncio
    async def test_records_experience(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        exp_id = await mem.record_experience(
            action_type="generation",
            summary="Generated a React component",
            outcome="success",
            score=0.9,
            agent_role="Builder",
            model="devstral-small",
            cost=0.002,
        )
        assert exp_id
        assert mem.experience_count == 1


class TestStats:
    """TramontaneMemory.stats()."""

    def test_stats_empty(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        stats = mem.stats()
        assert stats.fact_count == 0
        assert stats.experience_count == 0
        assert stats.working_block_count == 0

    def test_stats_after_inserts(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        import json

        mem._conn.execute(
            "INSERT INTO factual_memory (id, content, entity, embedding) VALUES (?, ?, ?, ?)",
            ("f1", "fact", "e", json.dumps([])),
        )
        mem._conn.execute(
            "INSERT INTO experiential_memory (id, action_type, action_summary) VALUES (?, ?, ?)",
            ("e1", "test", "test exp"),
        )
        mem._conn.commit()
        stats = mem.stats()
        assert stats.fact_count == 1
        assert stats.experience_count == 1


class TestFormatContext:
    """TramontaneMemory.format_context()."""

    def test_formats_results(self) -> None:
        results = [
            {"category": "fact", "content": "Paris is in France", "score": 0.95},
        ]
        ctx = TramontaneMemory.format_context(results)
        assert "Paris is in France" in ctx
        assert "0.95" in ctx

    def test_empty_results(self) -> None:
        assert TramontaneMemory.format_context([]) == ""


class TestImports:
    """Package-level imports."""

    def test_tramontane_memory_importable(self) -> None:
        from tramontane import TramontaneMemory
        assert TramontaneMemory is not None

    def test_working_block_importable(self) -> None:
        from tramontane import WorkingBlock
        assert WorkingBlock is not None

    def test_create_memory_tools_importable(self) -> None:
        from tramontane import create_memory_tools
        assert create_memory_tools is not None
