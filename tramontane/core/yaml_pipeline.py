"""YAML pipeline definitions — define pipelines in YAML, not just Python."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from tramontane.core.agent import Agent, AgentResult, RunContext

logger = logging.getLogger(__name__)


class AgentSpec(BaseModel):
    """Agent specification from YAML."""

    role: str
    goal: str
    backstory: str = ""
    model: str = "auto"
    temperature: float | None = None
    reasoning_effort: str | None = None
    reasoning_strategy: str = "fixed"
    max_tokens: int | None = None
    budget_eur: float | None = None
    routing_hint: str | None = None
    max_iter: int = 20

    def to_agent(self) -> Agent:
        """Convert spec to Agent instance."""
        kwargs: dict[str, Any] = {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "model": self.model,
            "max_iter": self.max_iter,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_strategy != "fixed":
            kwargs["reasoning_strategy"] = self.reasoning_strategy
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.budget_eur is not None:
            kwargs["budget_eur"] = self.budget_eur
        if self.routing_hint is not None:
            kwargs["routing_hint"] = self.routing_hint
        return Agent(**kwargs)


class PipelineSpec(BaseModel):
    """Pipeline specification from YAML."""

    name: str
    version: str = "1.0"
    description: str = ""
    budget_eur: float | None = None
    agents: dict[str, AgentSpec]
    flow: list[str]

    def validate_flow(self) -> list[str]:
        """Validate that all flow entries reference defined agents."""
        return [
            f"Flow references undefined agent: '{name}'"
            for name in self.flow
            if name not in self.agents
        ]


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    """Load a pipeline specification from a YAML file.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If flow references missing agents.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Pipeline YAML not found: {path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = PipelineSpec(**raw)

    errors = spec.validate_flow()
    if errors:
        msg = f"Pipeline validation failed: {'; '.join(errors)}"
        raise ValueError(msg)

    logger.info(
        "Loaded pipeline '%s' v%s with %d agents",
        spec.name,
        spec.version,
        len(spec.agents),
    )
    return spec


def create_agents_from_spec(spec: PipelineSpec) -> list[Agent]:
    """Create Agent instances from a pipeline spec, in flow order."""
    return [spec.agents[name].to_agent() for name in spec.flow]


async def run_yaml_pipeline(
    path: str | Path,
    input_text: str,
    *,
    router: Any | None = None,
) -> list[AgentResult]:
    """Load and run a YAML-defined pipeline.

    Agents execute sequentially, each receiving the previous agent's output.
    """
    spec = load_pipeline_spec(path)
    agents = create_agents_from_spec(spec)

    ctx = RunContext(budget_eur=spec.budget_eur) if spec.budget_eur else None

    results: list[AgentResult] = []
    current_input = input_text

    for agent in agents:
        result = await agent.run(
            current_input,
            router=router,
            run_context=ctx,
        )
        results.append(result)
        current_input = result.output
        logger.info(
            "YAML pipeline '%s': %s complete (model=%s, cost=EUR %.4f)",
            spec.name,
            agent.role,
            result.model_used,
            result.cost_eur,
        )

    total_cost = sum(r.cost_eur for r in results)
    logger.info(
        "YAML pipeline '%s' complete: %d agents, total cost=EUR %.4f",
        spec.name,
        len(results),
        total_cost,
    )

    return results
