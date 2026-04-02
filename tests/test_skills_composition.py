"""Tests for tramontane.skills.composition — pipelines + parallel."""

from __future__ import annotations

from typing import Any

import pytest

from tramontane.skills.base import Skill, SkillResult
from tramontane.skills.composition import (
    ConditionalSkill,
    ParallelSkills,
    SkillPersona,
    SkillPipeline,
)


class _Upper(Skill):
    name = "upper"
    description = "Uppercase text"

    async def execute(self, i: str, c: dict[str, Any] | None = None) -> SkillResult:
        return SkillResult(output=i.upper(), success=True)


class _Reverse(Skill):
    name = "reverse"
    description = "Reverse text"

    async def execute(self, i: str, c: dict[str, Any] | None = None) -> SkillResult:
        return SkillResult(output=i[::-1], success=True)


class _Echo(Skill):
    name = "echo"
    description = "Echo input"

    async def execute(self, i: str, c: dict[str, Any] | None = None) -> SkillResult:
        ctx_msg = (c or {}).get("persona", "")
        return SkillResult(output=f"{ctx_msg}{i}", success=True)


class TestSkillPipeline:
    @pytest.mark.asyncio
    async def test_sequential_execution(self) -> None:
        pipe = SkillPipeline([_Upper(), _Reverse()])
        results = await pipe.run("hello")
        assert len(results) == 2
        assert results[0].output == "HELLO"
        assert results[1].output == "OLLEH"

    @pytest.mark.asyncio
    async def test_with_persona(self) -> None:
        persona = SkillPersona(
            name="Gerald",
            description="Business agent",
            instructions="[Gerald] ",
        )
        pipe = SkillPipeline([_Echo()], persona=persona)
        results = await pipe.run("test")
        assert results[0].output == "[Gerald] test"


class TestConditionalSkill:
    @pytest.mark.asyncio
    async def test_runs_when_true(self) -> None:
        cond = ConditionalSkill(
            skill=_Upper(),
            condition=lambda prev: True,
        )
        pipe = SkillPipeline([cond])
        results = await pipe.run("hello")
        assert len(results) == 1
        assert results[0].output == "HELLO"

    @pytest.mark.asyncio
    async def test_skips_when_false(self) -> None:
        cond = ConditionalSkill(
            skill=_Upper(),
            condition=lambda prev: False,
        )
        pipe = SkillPipeline([cond])
        results = await pipe.run("hello")
        assert len(results) == 0


class TestParallelSkills:
    @pytest.mark.asyncio
    async def test_runs_concurrently(self) -> None:
        par = ParallelSkills([_Upper(), _Reverse()])
        results = await par.run("hello")
        assert len(results) == 2
        outputs = {r.output for r in results}
        assert "HELLO" in outputs
        assert "olleh" in outputs


class TestSkillPersona:
    def test_persona_fields(self) -> None:
        p = SkillPersona(
            name="Gerald",
            description="EU business agent",
            instructions="You are Gerald...",
            locale="fr",
            tone="professional",
        )
        assert p.name == "Gerald"
        assert p.locale == "fr"


class TestImports:
    def test_pipeline_importable(self) -> None:
        from tramontane import SkillPipeline
        assert SkillPipeline is not None

    def test_conditional_importable(self) -> None:
        from tramontane import ConditionalSkill
        assert ConditionalSkill is not None

    def test_parallel_importable(self) -> None:
        from tramontane import ParallelSkills
        assert ParallelSkills is not None
