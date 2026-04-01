"""FleetTuner — auto-discover optimal model + parameters for any agent."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from tramontane.core.agent import Agent
from tramontane.router.models import MISTRAL_MODELS

logger = logging.getLogger(__name__)


@dataclass
class TuneConfig:
    """A single configuration to test."""

    model: str
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass
class TuneResult:
    """Result from tuning a single configuration."""

    config: TuneConfig
    avg_cost_eur: float = 0.0
    avg_latency_s: float = 0.0
    avg_output_tokens: int = 0
    validation_pass_rate: float = 0.0
    total_cost_eur: float = 0.0
    num_prompts: int = 0
    errors: int = 0


@dataclass
class FleetTuneResult:
    """Complete tuning result with optimal config."""

    optimal: TuneConfig
    optimal_model: str = ""
    optimal_temperature: float | None = None
    optimal_reasoning_effort: str | None = None
    optimal_max_tokens: int | None = None
    avg_cost_eur: float = 0.0
    avg_quality_score: float = 0.0
    validation_pass_rate: float = 0.0
    tested_configs: int = 0
    total_tuning_cost_eur: float = 0.0
    savings_vs_default: str = ""
    all_results: list[TuneResult] = field(default_factory=list)

    def apply(self, agent: Agent) -> Agent:
        """Create a new Agent with the optimal config applied."""
        updates: dict[str, Any] = {"model": self.optimal.model}
        if self.optimal.reasoning_effort is not None:
            updates["reasoning_effort"] = self.optimal.reasoning_effort
        if self.optimal.temperature is not None:
            updates["temperature"] = self.optimal.temperature
        if self.optimal.max_tokens is not None:
            updates["max_tokens"] = self.optimal.max_tokens
        return agent.model_copy(update=updates)


class FleetTuner:
    """Auto-discover the optimal model + parameters for any agent role.

    Tests multiple configurations against real prompts and picks
    the Pareto-optimal setup based on the optimization target.

    Usage:
        tuner = FleetTuner()
        result = await tuner.tune(
            agent=builder,
            test_prompts=["Build a bakery page", "Build a SaaS page"],
            optimize_for="cost",
        )
        builder = result.apply(builder)
    """

    def __init__(
        self,
        models_to_test: list[str] | None = None,
        effort_levels: list[str | None] | None = None,
    ) -> None:
        """Initialize the tuner.

        Args:
            models_to_test: Models to include. Default: 4 common models.
            effort_levels: Reasoning effort levels to test.
        """
        self._models = models_to_test or [
            "ministral-3b",
            "mistral-small-4",
            "devstral-small",
            "devstral-2",
        ]
        self._efforts: list[str | None] = (
            effort_levels
            if effort_levels is not None
            else [None, "none", "medium", "high"]
        )

    def _generate_configs(self) -> list[TuneConfig]:
        """Generate all configs to test."""
        configs: list[TuneConfig] = []
        for model in self._models:
            model_info = MISTRAL_MODELS.get(model)
            supports_effort = bool(
                model_info and model_info.supports_reasoning_effort
            )

            if supports_effort:
                for effort in self._efforts:
                    if effort is not None:
                        configs.append(
                            TuneConfig(model=model, reasoning_effort=effort),
                        )
            else:
                configs.append(TuneConfig(model=model))

        return configs

    async def _test_config(
        self,
        agent: Agent,
        config: TuneConfig,
        prompts: list[str],
    ) -> TuneResult:
        """Test a single config against all prompts."""
        test_agent = agent.model_copy(
            update={
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
        )

        result = TuneResult(config=config, num_prompts=len(prompts))
        costs: list[float] = []
        latencies: list[float] = []
        output_tokens: list[int] = []
        validations_passed = 0

        for prompt in prompts:
            try:
                t0 = time.monotonic()
                agent_result = await test_agent.run(prompt)
                elapsed = time.monotonic() - t0

                costs.append(agent_result.cost_eur)
                latencies.append(elapsed)
                output_tokens.append(agent_result.output_tokens)

                if agent.validate_output is None or agent.validate_output(
                    agent_result,
                ):
                    validations_passed += 1

            except Exception as exc:
                logger.warning(
                    "FleetTuner: %s failed on prompt: %s",
                    config.model,
                    str(exc)[:100],
                )
                result.errors += 1

        if costs:
            result.avg_cost_eur = sum(costs) / len(costs)
            result.avg_latency_s = sum(latencies) / len(latencies)
            result.avg_output_tokens = int(sum(output_tokens) / len(output_tokens))
            result.total_cost_eur = sum(costs)

        successful = len(prompts) - result.errors
        if successful > 0:
            result.validation_pass_rate = validations_passed / successful

        return result

    async def tune(
        self,
        agent: Agent,
        test_prompts: list[str],
        optimize_for: Literal["cost", "quality", "balanced", "speed"] = "balanced",
    ) -> FleetTuneResult:
        """Run the tuning process.

        Args:
            agent: The agent to optimize.
            test_prompts: 3-5 representative prompts.
            optimize_for: "cost", "quality", "balanced", or "speed".

        Returns:
            FleetTuneResult with the optimal configuration.
        """
        configs = self._generate_configs()
        logger.info(
            "FleetTuner: Testing %d configs against %d prompts for agent '%s'",
            len(configs),
            len(test_prompts),
            agent.role,
        )

        all_results: list[TuneResult] = []
        for i, config in enumerate(configs):
            logger.info(
                "FleetTuner: Testing config %d/%d: %s (effort=%s)",
                i + 1,
                len(configs),
                config.model,
                config.reasoning_effort,
            )
            tune_result = await self._test_config(agent, config, test_prompts)
            all_results.append(tune_result)

        best = self._select_optimal(all_results, optimize_for)

        default_cost = all_results[0].avg_cost_eur if all_results else 0
        if default_cost > 0 and best.avg_cost_eur > 0:
            savings_pct = (1 - best.avg_cost_eur / default_cost) * 100
            savings_str = f"{savings_pct:.0f}%"
        else:
            savings_str = "N/A"

        total_tuning_cost = sum(r.total_cost_eur for r in all_results)

        return FleetTuneResult(
            optimal=best.config,
            optimal_model=best.config.model,
            optimal_temperature=best.config.temperature,
            optimal_reasoning_effort=best.config.reasoning_effort,
            optimal_max_tokens=best.config.max_tokens,
            avg_cost_eur=best.avg_cost_eur,
            avg_quality_score=best.validation_pass_rate,
            validation_pass_rate=best.validation_pass_rate,
            tested_configs=len(configs),
            total_tuning_cost_eur=total_tuning_cost,
            savings_vs_default=savings_str,
            all_results=all_results,
        )

    @staticmethod
    def _select_optimal(
        results: list[TuneResult],
        optimize_for: str,
    ) -> TuneResult:
        """Select the best config based on optimization target."""
        viable = [r for r in results if r.errors < r.num_prompts]
        if not viable:
            viable = results

        if optimize_for == "cost":
            qualified = [r for r in viable if r.validation_pass_rate >= 0.7]
            if qualified:
                return min(qualified, key=lambda r: r.avg_cost_eur)
            return max(viable, key=lambda r: r.validation_pass_rate)

        if optimize_for == "quality":
            return max(viable, key=lambda r: r.validation_pass_rate)

        if optimize_for == "speed":
            qualified = [r for r in viable if r.validation_pass_rate >= 0.7]
            if qualified:
                return min(qualified, key=lambda r: r.avg_latency_s)
            return max(viable, key=lambda r: r.validation_pass_rate)

        # balanced
        max_cost = max((r.avg_cost_eur for r in viable), default=1) or 1
        max_latency = max((r.avg_latency_s for r in viable), default=1) or 1

        def score(r: TuneResult) -> float:
            quality = r.validation_pass_rate
            cost_norm = 1 - (r.avg_cost_eur / max_cost)
            speed_norm = 1 - (r.avg_latency_s / max_latency)
            return quality * 0.5 + cost_norm * 0.3 + speed_norm * 0.2

        return max(viable, key=score)
