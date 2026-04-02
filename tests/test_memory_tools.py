"""Tests for tramontane.memory.tools — agent-callable memory tools."""

from __future__ import annotations

from tramontane.core.agent import Agent
from tramontane.memory.engine import TramontaneMemory
from tramontane.memory.tools import create_memory_tools


class TestCreateMemoryTools:
    """create_memory_tools() factory."""

    def test_returns_five_functions(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        tools = create_memory_tools(mem)
        assert len(tools) == 5

    def test_all_are_callable(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        tools = create_memory_tools(mem)
        for tool in tools:
            assert callable(tool)

    def test_tool_names(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        tools = create_memory_tools(mem)
        names = {t.__name__ for t in tools}
        assert names == {
            "retain_memory", "recall_memory", "reflect_on_memory",
            "forget_memory", "update_memory",
        }

    def test_tools_have_docstrings(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        tools = create_memory_tools(mem)
        for tool in tools:
            assert tool.__doc__, f"{tool.__name__} missing docstring"


class TestAgentMemoryTools:
    """Agent with memory_tools=True."""

    def test_agent_accepts_memory_tools_field(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            memory_tools=True,
        )
        assert a.memory_tools is True

    def test_agent_accepts_tramontane_memory(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/mem.db"  # type: ignore[operator]
        mem = TramontaneMemory(db_path=db)
        a = Agent(
            role="R", goal="G", backstory="B",
            tramontane_memory=mem,
            memory_tools=True,
        )
        assert a.tramontane_memory is mem
