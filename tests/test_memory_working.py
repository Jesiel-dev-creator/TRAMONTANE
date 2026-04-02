"""Tests for tramontane.memory.working — working memory blocks."""

from __future__ import annotations

from tramontane.memory.engine import TramontaneMemory
from tramontane.memory.working import WorkingMemoryManager


class TestWorkingMemory:
    """WorkingMemoryManager operations."""

    def test_set_and_get_block(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)

        wm.set_block("agent1", "Goals", "Build a website")
        blocks = wm.get_blocks("agent1")
        assert len(blocks) == 1
        assert blocks[0].label == "Goals"
        assert blocks[0].content == "Build a website"

    def test_update_existing_block(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)

        wm.set_block("agent1", "Goals", "v1")
        wm.set_block("agent1", "Goals", "v2")
        blocks = wm.get_blocks("agent1")
        assert len(blocks) == 1
        assert blocks[0].content == "v2"

    def test_delete_block(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)

        wm.set_block("agent1", "Temp", "data")
        wm.delete_block("agent1", "Temp")
        assert wm.get_blocks("agent1") == []

    def test_format_for_prompt(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)

        wm.set_block("agent1", "Goals", "Build a bakery site")
        wm.set_block("agent1", "User", "Name: Alice, Pref: warm colors")

        prompt = wm.format_for_prompt("agent1")
        assert "Working Memory" in prompt
        assert "Goals" in prompt
        assert "Build a bakery site" in prompt
        assert "User" in prompt

    def test_format_empty(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)
        assert wm.format_for_prompt("nonexistent") == ""

    def test_agent_isolation(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        wm = WorkingMemoryManager(mem._conn)

        wm.set_block("agent1", "Data", "agent1 data")
        wm.set_block("agent2", "Data", "agent2 data")

        blocks1 = wm.get_blocks("agent1")
        blocks2 = wm.get_blocks("agent2")
        assert len(blocks1) == 1
        assert len(blocks2) == 1
        assert blocks1[0].content == "agent1 data"
        assert blocks2[0].content == "agent2 data"
