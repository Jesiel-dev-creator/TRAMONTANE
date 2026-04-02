"""Tests for tramontane.skills.base — Skill ABC + decorators."""

from __future__ import annotations

from typing import Any

import pytest

from tramontane.skills.base import Skill, SkillResult, track_skill


class SimpleSkill(Skill):
    name = "simple"
    description = "A simple test skill"
    version = "1.0"
    triggers = ["do it", "make it"]
    preferred_model = "mistral-small"
    memory_tags = ["test"]
    tags = ["test"]

    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(output=f"done: {input_text}", success=True)


class FailingSkill(Skill):
    name = "failing"
    description = "Always fails"

    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        msg = "intentional failure"
        raise RuntimeError(msg)


class TestSkillMetadata:
    def test_has_name(self) -> None:
        s = SimpleSkill()
        assert s.name == "simple"

    def test_has_triggers(self) -> None:
        s = SimpleSkill()
        assert "do it" in s.triggers

    def test_to_dict(self) -> None:
        s = SimpleSkill()
        d = s.to_dict()
        assert d["name"] == "simple"
        assert d["triggers"] == ["do it", "make it"]
        assert d["memory_tags"] == ["test"]

    def test_to_mcp_tool(self) -> None:
        s = SimpleSkill()
        mcp = s.to_mcp_tool()
        assert mcp["name"] == "simple"
        assert "inputSchema" in mcp
        assert mcp["inputSchema"]["properties"]["input_text"]["type"] == "string"


class TestSkillMatches:
    def test_exact_trigger(self) -> None:
        assert SimpleSkill().matches("do it now") == 1.0

    def test_name_match(self) -> None:
        assert SimpleSkill().matches("use simple skill") == 0.8

    def test_partial_description(self) -> None:
        score = SimpleSkill().matches("test something")
        assert 0 < score <= 0.6

    def test_no_match(self) -> None:
        assert SimpleSkill().matches("xyz unrelated") == 0.0


class TestSkillResult:
    def test_defaults(self) -> None:
        r = SkillResult()
        assert r.output == ""
        assert r.success is False
        assert r.error is None

    def test_with_values(self) -> None:
        r = SkillResult(output="hello", success=True, cost_eur=0.001)
        assert r.output == "hello"
        assert r.cost_eur == 0.001


class TestTrackSkillDecorator:
    @pytest.mark.asyncio
    async def test_tracks_timing(self) -> None:
        class TrackedSkill(Skill):
            name = "tracked"
            description = "test"

            @track_skill
            async def execute(self, input_text: str, context: Any = None) -> SkillResult:
                return SkillResult(output="ok", success=True)

        result = await TrackedSkill().execute("test")
        assert "duration_s" in result.metadata
        assert result.metadata["skill_name"] == "tracked"

    @pytest.mark.asyncio
    async def test_catches_exceptions(self) -> None:
        class BadSkill(Skill):
            name = "bad"
            description = "test"

            @track_skill
            async def execute(self, input_text: str, context: Any = None) -> SkillResult:
                msg = "boom"
                raise RuntimeError(msg)

        result = await BadSkill().execute("test")
        assert result.success is False
        assert result.error == "boom"


class TestExecuteWithMemory:
    @pytest.mark.asyncio
    async def test_without_memory(self) -> None:
        s = SimpleSkill()
        result = await s.execute_with_memory("test", memory=None)
        assert result.success is True
        assert result.output == "done: test"


class TestValidate:
    def test_default_validation(self) -> None:
        s = SimpleSkill()
        assert s.validate(SkillResult(output="hello", success=True)) is True
        assert s.validate(SkillResult(output="", success=True)) is False


class TestImports:
    def test_skill_importable(self) -> None:
        from tramontane import Skill
        assert Skill is not None

    def test_skill_result_importable(self) -> None:
        from tramontane import SkillResult
        assert SkillResult is not None
