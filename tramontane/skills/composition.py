"""Skill composition — chain skills into pipelines."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from tramontane.skills.base import Skill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class SkillPersona:
    """Define personality/behavior context for skill execution.

    Inspired by OpenClaw SOUL.md. Separates WHO the agent is
    from WHAT the agent can do.
    """

    name: str
    description: str
    instructions: str
    locale: str = "en"
    tone: str = "professional"


@dataclass
class ConditionalSkill:
    """Wrapper that only executes if condition is met."""

    skill: Skill
    condition: Callable[[SkillResult | None], bool]

    def should_run(self, previous_result: SkillResult | None) -> bool:
        """Check if this skill should execute."""
        return self.condition(previous_result)


class SkillPipeline:
    """Chain skills sequentially. Output of each feeds into next."""

    def __init__(
        self,
        skills: list[Skill | ConditionalSkill],
        persona: SkillPersona | None = None,
    ) -> None:
        self._skills = skills
        self._persona = persona

    async def run(
        self,
        input_text: str,
        context: dict[str, Any] | None = None,
    ) -> list[SkillResult]:
        """Execute skills in sequence."""
        results: list[SkillResult] = []
        current_input = input_text
        ctx = dict(context) if context else {}

        if self._persona:
            ctx["persona"] = self._persona.instructions

        for entry in self._skills:
            if isinstance(entry, ConditionalSkill):
                prev = results[-1] if results else None
                if not entry.should_run(prev):
                    continue
                result = await entry.skill.execute(current_input, ctx)
            else:
                result = await entry.execute(current_input, ctx)

            results.append(result)
            current_input = result.output
            ctx["previous_results"] = results

        return results


class ParallelSkills:
    """Run multiple skills concurrently on the same input."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = skills

    async def run(
        self,
        input_text: str,
        context: dict[str, Any] | None = None,
    ) -> list[SkillResult]:
        """Execute all skills in parallel."""
        tasks = [s.execute(input_text, context) for s in self._skills]
        results: list[SkillResult] = await asyncio.gather(*tasks)
        return results
