"""Parallel agent execution — run independent agents simultaneously."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from tramontane.core.agent import Agent, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class ParallelResult:
    """Result from running agents in parallel."""

    results: dict[str, AgentResult] = field(default_factory=dict)
    total_cost_eur: float = 0.0
    total_duration_s: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)

    def merge(self, separator: str = "\n\n---\n\n") -> str:
        """Merge all outputs into a single string."""
        return separator.join(
            r.output for r in self.results.values() if r.output
        )

    def get(self, role: str) -> AgentResult | None:
        """Get result by agent role."""
        return self.results.get(role)


class ParallelGroup:
    """A group of agents that run concurrently.

    Usage:
        group = ParallelGroup([designer, architect])
        result = await group.run(
            input_text="Design a bakery website",
            router=router,
        )
        designer_output = result.get("Designer")
        architect_output = result.get("Architect")
    """

    def __init__(
        self,
        agents: list[Agent],
        merge_fn: Callable[[dict[str, AgentResult]], str] | None = None,
    ) -> None:
        """Initialize parallel group.

        Args:
            agents: Agents to run in parallel.
            merge_fn: Optional custom merge function.
        """
        self._agents = agents
        self._merge_fn = merge_fn

    async def run(
        self,
        input_text: str | None = None,
        inputs: dict[str, str] | None = None,
        *,
        router: Any | None = None,
        run_context: Any | None = None,
    ) -> ParallelResult:
        """Run all agents in parallel.

        Args:
            input_text: Shared input for all agents.
            inputs: Per-agent inputs keyed by role. Priority over input_text.
            router: MistralRouter for model="auto" agents.
            run_context: RunContext for shared budget tracking.

        Returns:
            ParallelResult with individual and merged results.
        """
        start = time.monotonic()

        async def _run_one(
            agent: Agent,
        ) -> tuple[str, AgentResult | None, str | None]:
            agent_input = (
                inputs.get(agent.role, input_text or "")
                if inputs
                else (input_text or "")
            )
            try:
                result = await agent.run(
                    agent_input,
                    router=router,
                    run_context=run_context,
                )
                return (agent.role, result, None)
            except Exception as exc:
                logger.error("Parallel agent '%s' failed: %s", agent.role, exc)
                return (agent.role, None, str(exc))

        tasks = [_run_one(agent) for agent in self._agents]
        completed = await asyncio.gather(*tasks)

        result = ParallelResult()
        for role, agent_result, error in completed:
            if agent_result:
                result.results[role] = agent_result
                result.total_cost_eur += agent_result.cost_eur
            if error:
                result.errors[role] = error

        result.total_duration_s = round(time.monotonic() - start, 2)
        return result

    @property
    def agents(self) -> list[Agent]:
        """The agents in this group."""
        return self._agents
