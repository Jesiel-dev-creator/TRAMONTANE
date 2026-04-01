"""Tests for tramontane.core.tuner — FleetTuner."""

from __future__ import annotations

from tramontane.core.agent import Agent
from tramontane.core.tuner import (
    FleetTuner,
    FleetTuneResult,
    TuneConfig,
    TuneResult,
)


class TestGenerateConfigs:
    """FleetTuner._generate_configs() configuration generation."""

    def test_default_models_generate_configs(self) -> None:
        tuner = FleetTuner()
        configs = tuner._generate_configs()
        # ministral-3b (1) + mistral-small-4 (3 efforts) + devstral-small (1) + devstral-2 (1)
        assert len(configs) == 6

    def test_effort_levels_for_supporting_model(self) -> None:
        tuner = FleetTuner(models_to_test=["mistral-small-4"])
        configs = tuner._generate_configs()
        # 3 effort levels: none, medium, high (None skipped for supporting models)
        assert len(configs) == 3
        efforts = {c.reasoning_effort for c in configs}
        assert efforts == {"none", "medium", "high"}

    def test_non_supporting_model_single_config(self) -> None:
        tuner = FleetTuner(models_to_test=["devstral-small"])
        configs = tuner._generate_configs()
        assert len(configs) == 1
        assert configs[0].reasoning_effort is None

    def test_custom_models(self) -> None:
        tuner = FleetTuner(models_to_test=["ministral-3b", "ministral-7b"])
        configs = tuner._generate_configs()
        assert len(configs) == 2


class TestSelectOptimal:
    """FleetTuner._select_optimal() selection logic."""

    def _make_result(
        self,
        model: str,
        cost: float,
        quality: float,
        latency: float,
        errors: int = 0,
    ) -> TuneResult:
        return TuneResult(
            config=TuneConfig(model=model),
            avg_cost_eur=cost,
            validation_pass_rate=quality,
            avg_latency_s=latency,
            num_prompts=5,
            errors=errors,
        )

    def test_cost_picks_cheapest_viable(self) -> None:
        results = [
            self._make_result("expensive", 0.01, 0.9, 2.0),
            self._make_result("cheap", 0.001, 0.8, 1.0),
            self._make_result("cheapest-bad", 0.0001, 0.3, 0.5),
        ]
        best = FleetTuner._select_optimal(results, "cost")
        assert best.config.model == "cheap"  # cheapest with >70% quality

    def test_quality_picks_highest_pass_rate(self) -> None:
        results = [
            self._make_result("ok", 0.001, 0.7, 1.0),
            self._make_result("best", 0.01, 0.95, 3.0),
            self._make_result("mid", 0.005, 0.8, 2.0),
        ]
        best = FleetTuner._select_optimal(results, "quality")
        assert best.config.model == "best"

    def test_speed_picks_fastest_viable(self) -> None:
        results = [
            self._make_result("slow", 0.01, 0.9, 5.0),
            self._make_result("fast", 0.005, 0.8, 0.5),
            self._make_result("fastest-bad", 0.001, 0.3, 0.1),
        ]
        best = FleetTuner._select_optimal(results, "speed")
        assert best.config.model == "fast"

    def test_balanced_uses_weighted_score(self) -> None:
        results = [
            self._make_result("expensive-good", 0.01, 1.0, 3.0),
            self._make_result("balanced", 0.003, 0.9, 1.0),
            self._make_result("cheap-bad", 0.001, 0.5, 0.5),
        ]
        best = FleetTuner._select_optimal(results, "balanced")
        # "balanced" should score well: decent quality + low cost + fast
        assert best.config.model == "balanced"

    def test_all_errors_returns_best_of_bad(self) -> None:
        results = [
            self._make_result("a", 0.01, 0.0, 1.0, errors=5),
            self._make_result("b", 0.001, 0.0, 0.5, errors=5),
        ]
        # Both have all errors — should still pick one
        best = FleetTuner._select_optimal(results, "cost")
        assert best is not None


class TestFleetTuneResultApply:
    """FleetTuneResult.apply() creates optimized agent."""

    def test_apply_creates_new_agent(self) -> None:
        agent = Agent(role="R", goal="G", backstory="B", model="auto")
        result = FleetTuneResult(
            optimal=TuneConfig(
                model="devstral-small",
                reasoning_effort="medium",
                temperature=0.3,
            ),
        )
        new_agent = result.apply(agent)
        assert new_agent.model == "devstral-small"
        assert new_agent.reasoning_effort == "medium"
        assert new_agent.temperature == 0.3
        # Original unchanged
        assert agent.model == "auto"

    def test_apply_preserves_role(self) -> None:
        agent = Agent(role="Builder", goal="Build", backstory="Expert")
        result = FleetTuneResult(
            optimal=TuneConfig(model="mistral-small-4"),
        )
        new_agent = result.apply(agent)
        assert new_agent.role == "Builder"
        assert new_agent.model == "mistral-small-4"


class TestImports:
    """Package-level imports."""

    def test_fleet_tuner_importable(self) -> None:
        from tramontane import FleetTuner

        assert FleetTuner is not None

    def test_fleet_tune_result_importable(self) -> None:
        from tramontane import FleetTuneResult

        assert FleetTuneResult is not None
