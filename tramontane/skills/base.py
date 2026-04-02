"""TramontaneSkills — typed, composable, learnable skill system.

Skills are reusable capabilities that agents can discover and execute.
Unlike tools (single functions), skills combine: an agent configuration,
tools, prompts, memory context, validation, and model preferences.
"""

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result from executing a skill."""

    output: str = ""
    parsed_output: Any | None = None
    cost_eur: float = 0.0
    model_used: str = ""
    success: bool = False
    validation_passed: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def track_skill(
    func: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator that profiles skill execution automatically.

    Inspired by NVIDIA NeMo @track_function. Tracks timing, cost,
    success/failure. Catches exceptions and returns SkillResult(error=...).
    """

    @functools.wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> SkillResult:
        start = time.monotonic()
        skill_name = getattr(self, "name", func.__name__)

        try:
            result: SkillResult = await func(self, *args, **kwargs)
            duration = time.monotonic() - start
            result.metadata["duration_s"] = round(duration, 3)
            result.metadata["skill_name"] = skill_name
            result.metadata["skill_version"] = getattr(self, "version", "?")
            logger.info(
                "Skill '%s' completed: success=%s cost=EUR %.4f dur=%.2fs",
                skill_name, result.success, result.cost_eur, duration,
            )
            return result

        except Exception as exc:
            duration = time.monotonic() - start
            logger.error(
                "Skill '%s' failed after %.2fs: %s", skill_name, duration, exc,
            )
            return SkillResult(
                success=False,
                error=str(exc),
                metadata={
                    "duration_s": round(duration, 3),
                    "skill_name": skill_name,
                },
            )

    return wrapper


class Skill(ABC):
    """Base class for all Tramontane skills.

    Subclass this to create a skill:

        class MySkill(Skill):
            name = "my_skill"
            description = "Does something useful"
            triggers = ["do the thing"]
            async def execute(self, input_text, context=None):
                return SkillResult(output="done", success=True)
    """

    name: str = ""
    description: str = ""
    version: str = "1.0"
    triggers: list[str] = []
    preferred_model: str = "auto"
    preferred_temperature: float | None = None
    max_tokens: int | None = None
    budget_eur: float | None = None
    tools: list[Callable[..., Any]] = []
    output_schema: type[BaseModel] | None = None
    memory_tags: list[str] = []
    author: str = ""
    tags: list[str] = []

    @abstractmethod
    async def execute(
        self,
        input_text: str,
        context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Execute the skill."""
        ...

    def validate(self, result: SkillResult) -> bool:
        """Validate the skill's output."""
        if self.output_schema and result.parsed_output:
            try:
                self.output_schema.model_validate(result.parsed_output)
                return True
            except Exception:
                return False
        return bool(result.output)

    def matches(self, query: str) -> float:
        """Score how well this skill matches a query. 0-1."""
        query_lower = query.lower()
        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                return 1.0
        if self.name.lower() in query_lower:
            return 0.8
        query_words = set(query_lower.split())
        desc_words = set(self.description.lower().split())
        overlap = len(query_words & desc_words)
        if overlap > 0:
            return min(overlap / max(len(query_words), 1) * 0.6, 0.6)
        return 0.0

    async def execute_with_memory(
        self,
        input_text: str,
        memory: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Execute skill with memory integration.

        Before: recalls relevant memories using memory_tags.
        After: records experience for learning.
        """
        ctx = dict(context) if context else {}

        if memory and self.memory_tags:
            tag_query = " ".join(self.memory_tags)
            memories = await memory.recall(tag_query, top_k=3)
            if memories:
                ctx["memory_context"] = memory.format_context(memories)

        result = await self.execute(input_text, ctx)

        if memory and result.success:
            await memory.record_experience(
                action_type=self.name,
                action_summary=f"{self.name}: {input_text[:200]}",
                outcome=result.output[:500] if result.output else "",
                outcome_score=1.0 if result.validation_passed else 0.5,
                agent_role=self.name,
                model_used=result.model_used,
                cost_eur=result.cost_eur,
            )

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize skill metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "triggers": self.triggers,
            "preferred_model": self.preferred_model,
            "memory_tags": self.memory_tags,
            "author": self.author,
            "tags": self.tags,
        }

    def to_mcp_tool(self) -> dict[str, Any]:
        """Convert to MCP tool definition for publishing."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "The task input",
                    },
                },
                "required": ["input_text"],
            },
        }
