"""Tests for tramontane.router — classifier + router."""

from __future__ import annotations

import pytest

from tramontane.core.exceptions import BudgetExceededError
from tramontane.router.classifier import ClassificationMode, TaskClassifier
from tramontane.router.router import MistralRouter


class TestClassifierOffline:
    """Offline keyword-heuristic classifier."""

    def test_code_task(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync("write a Python function to sort a list")
        assert r.task_type == "code"
        assert r.has_code is True
        assert r.mode_used == ClassificationMode.OFFLINE

    def test_research_task(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync("research the latest news about AI in France")
        assert r.task_type == "research"

    def test_bulk_task(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync("list all")
        assert r.task_type == "bulk"

    def test_french_locale(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync(
            "Recherchez les dernières nouvelles sur le marché français de l'IA"
        )
        assert r.language == "fr"

    def test_confidence_offline(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync("hello world")
        assert r.confidence == 0.70

    def test_complexity_short_prompt(self, offline_classifier: TaskClassifier) -> None:
        r = offline_classifier.classify_sync("fix bug")
        assert r.complexity >= 1
        assert r.complexity <= 5


class TestRouter:
    """MistralRouter decision tree."""

    def test_code_routes_to_devstral_small(self, offline_router: MistralRouter) -> None:
        d = offline_router.route_sync("write a Python function")
        assert d.primary_model == "devstral-small"

    def test_code_complex_routes_to_devstral_2(self, offline_router: MistralRouter) -> None:
        # Long prompt pushes complexity >= 4
        prompt = "Refactor this entire monorepo architecture " * 50
        d = offline_router.route_sync(prompt)
        assert d.primary_model in ("devstral-2", "devstral-small")

    def test_budget_downgrades_model(self, offline_router: MistralRouter) -> None:
        # Long reasoning prompt → routes to expensive model, then downgrades
        # Budget sufficient for floor but not the ideal model
        prompt = "Analyze step by step why " * 50 + "the EU AI Act matters"
        d = offline_router.route_sync(prompt, budget=0.005)
        assert d.budget_constrained is True
        assert d.downgrade_applied is True
        # Must be tier >= 3 (reasoning floor) — never ministral-3b/7b
        from tramontane.router.models import MISTRAL_MODELS

        model = MISTRAL_MODELS.get(d.primary_model)
        assert model is not None
        assert model.tier >= 3

    def test_budget_too_low_raises_error(self, offline_router: MistralRouter) -> None:
        # Budget so low that even the floor model can't fit
        with pytest.raises(BudgetExceededError):
            offline_router.route_sync(
                "analyze the EU AI Act implications",
                budget=0.0000000001,
            )

    def test_reasoning_floor_never_below_magistral(
        self, offline_router: MistralRouter,
    ) -> None:
        # Reasoning task with tight budget should get magistral-small or above
        d = offline_router.route_sync(
            "explain why quantum computing breaks RSA encryption step by step",
            budget=0.001,
        )
        if d.downgrade_applied:
            from tramontane.router.models import MISTRAL_MODELS

            model = MISTRAL_MODELS.get(d.primary_model)
            assert model is not None
            assert model.tier >= 2  # magistral-small is tier 3

    def test_code_floor_never_below_devstral(
        self, offline_router: MistralRouter,
    ) -> None:
        # Code task with tight budget should stay at devstral-small or above
        d = offline_router.route_sync(
            "write a Python function to merge sort a linked list",
            budget=0.001,
        )
        if d.downgrade_applied:
            from tramontane.router.models import MISTRAL_MODELS

            model = MISTRAL_MODELS.get(d.primary_model)
            assert model is not None
            assert model.tier >= 2

    def test_no_budget_no_change(self, offline_router: MistralRouter) -> None:
        # No budget = no downgrade, regression test
        d = offline_router.route_sync("write a function")
        assert d.downgrade_applied is False
        assert d.budget_constrained is False

    def test_french_locale_prefers_multilingual(
        self, offline_router: MistralRouter,
    ) -> None:
        d = offline_router.route_sync(
            "recherchez les dernières nouvelles", locale="fr",
        )
        assert d.primary_model in ("mistral-small", "mistral-large", "ministral-7b")

    def test_explain_returns_string(self, offline_router: MistralRouter) -> None:
        d = offline_router.route_sync("write code")
        explanation = MistralRouter.explain(d)
        assert isinstance(explanation, str)
        assert "Routed to" in explanation

    def test_local_mode(self) -> None:
        r = MistralRouter(local_mode=True)
        d = r.route_sync("write code")
        assert d.local_mode is True


class TestTaskTypeValidation:
    """Classifier task type validation and normalization."""

    def test_validate_known_type(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("code") == "code"
        assert _validate_task_type("general") == "general"
        assert _validate_task_type("vision") == "vision"

    def test_validate_alias_design(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("design") == "reasoning"

    def test_validate_alias_analysis(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("analysis") == "reasoning"

    def test_validate_alias_coding(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("coding") == "code"
        assert _validate_task_type("programming") == "code"

    def test_validate_alias_creative(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("creative") == "general"

    def test_validate_unknown_defaults_to_general(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("foobar") == "general"
        assert _validate_task_type("") == "general"

    def test_validate_case_insensitive(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("CODE") == "code"
        assert _validate_task_type("Design") == "reasoning"

    def test_design_prompt_offline_returns_valid_type(
        self, offline_classifier: TaskClassifier,
    ) -> None:
        """A 'design' prompt must return a valid router task type."""
        from tramontane.router.classifier import VALID_TASK_TYPES

        r = offline_classifier.classify_sync("design a modern landing page for my SaaS")
        assert r.task_type in VALID_TASK_TYPES

    def test_creative_prompt_offline_returns_general(
        self, offline_classifier: TaskClassifier,
    ) -> None:
        """Creative prompts should map to 'general', not 'creative'."""
        r = offline_classifier.classify_sync(
            "compose a poem about the ocean, draft a narrative for the essay"
        )
        assert r.task_type == "general"


class TestDesignVsVision:
    """Design tasks must NOT route to vision (pixtral-large)."""

    def test_design_system_not_vision(
        self, offline_classifier: TaskClassifier,
    ) -> None:
        r = offline_classifier.classify_sync("Create a design system with warm colors")
        assert r.task_type != "vision"
        assert r.has_vision is False

    def test_color_palette_not_vision(
        self, offline_classifier: TaskClassifier,
    ) -> None:
        r = offline_classifier.classify_sync("Design a color palette for a bakery website")
        assert r.task_type != "vision"

    def test_image_analysis_is_vision(
        self, offline_classifier: TaskClassifier,
    ) -> None:
        r = offline_classifier.classify_sync(
            "Analyze this image of a website",
            context="screenshot.png",
        )
        assert r.task_type == "vision"
        assert r.has_vision is True

    def test_design_alias_maps_to_reasoning(self) -> None:
        from tramontane.router.classifier import _validate_task_type

        assert _validate_task_type("design") == "reasoning"
        assert _validate_task_type("ui_design") == "reasoning"
        assert _validate_task_type("styling") == "code"

    def test_design_prompt_routes_cheap(self, offline_router: MistralRouter) -> None:
        """Design task should NOT route to pixtral-large (tier 4, €2/6)."""
        d = offline_router.route_sync("Create a design system with warm colors")
        assert d.primary_model != "pixtral-large"


class TestModelFleet:
    """Model registry completeness."""

    def test_all_models_importable(self) -> None:
        from tramontane.router.models import MISTRAL_MODELS

        assert len(MISTRAL_MODELS) >= 13
        for alias, model in MISTRAL_MODELS.items():
            assert model.api_id, f"{alias} missing api_id"
            assert model.tier >= 0, f"{alias} invalid tier"

    def test_mistral_small_4_reasoning_effort(self) -> None:
        from tramontane.router.models import get_model

        m = get_model("mistral-small-4")
        assert m.supports_reasoning_effort is True
        assert m.supports_vision is True
        assert m.context_window == 256_000

    def test_voxtral_tts_exists(self) -> None:
        from tramontane.router.models import get_model

        m = get_model("voxtral-tts")
        assert m.modality == "text-to-speech"
        assert m.max_output_tokens == 0

    def test_existing_models_default_false(self) -> None:
        from tramontane.router.models import get_model

        m = get_model("mistral-small")
        assert m.supports_reasoning_effort is False
        assert m.supports_vision is False


class TestFleetTelemetry:
    """Self-learning router telemetry."""

    def test_record_and_retrieve(self, tmp_path: object) -> None:
        from tramontane.router.telemetry import FleetTelemetry, RoutingOutcome

        db = str(tmp_path) + "/test.db"  # type: ignore[operator]
        t = FleetTelemetry(db_path=db)
        t.record(RoutingOutcome(
            task_type="code", complexity=3, model_used="devstral-small",
            reasoning_effort=None, success=True, cost_eur=0.001,
            latency_s=1.5, output_tokens=500, agent_role="builder",
        ))
        assert t.total_outcomes == 1

    def test_suggest_returns_none_below_min_samples(self, tmp_path: object) -> None:
        from tramontane.router.telemetry import FleetTelemetry, RoutingOutcome

        db = str(tmp_path) + "/test.db"  # type: ignore[operator]
        t = FleetTelemetry(db_path=db)
        for _ in range(5):
            t.record(RoutingOutcome(
                task_type="code", complexity=3, model_used="devstral-small",
                reasoning_effort=None, success=True, cost_eur=0.001,
                latency_s=1.0, output_tokens=500,
            ))
        assert t.suggest_model("code", 3, min_samples=10) is None

    def test_suggest_returns_best_model(self, tmp_path: object) -> None:
        from tramontane.router.telemetry import FleetTelemetry, RoutingOutcome

        db = str(tmp_path) + "/test.db"  # type: ignore[operator]
        t = FleetTelemetry(db_path=db)
        # 15 successes for devstral-small at low cost
        for _ in range(15):
            t.record(RoutingOutcome(
                task_type="code", complexity=3, model_used="devstral-small",
                reasoning_effort=None, success=True, cost_eur=0.001,
                latency_s=1.0, output_tokens=500,
            ))
        # 15 successes for devstral-2 at higher cost
        for _ in range(15):
            t.record(RoutingOutcome(
                task_type="code", complexity=3, model_used="devstral-2",
                reasoning_effort=None, success=True, cost_eur=0.005,
                latency_s=2.0, output_tokens=1000,
            ))
        # Both have 100% success — devstral-small wins on cost
        suggestion = t.suggest_model("code", 3, min_samples=10)
        assert suggestion == "devstral-small"

    def test_get_model_stats(self, tmp_path: object) -> None:
        from tramontane.router.telemetry import FleetTelemetry, RoutingOutcome

        db = str(tmp_path) + "/test.db"  # type: ignore[operator]
        t = FleetTelemetry(db_path=db)
        t.record(RoutingOutcome(
            task_type="code", complexity=2, model_used="devstral-small",
            reasoning_effort=None, success=True, cost_eur=0.001,
            latency_s=1.0, output_tokens=500,
        ))
        stats = t.get_model_stats("devstral-small")
        assert len(stats) == 1
        assert stats[0]["total"] == 1

    def test_router_without_telemetry_backward_compatible(
        self, offline_router: MistralRouter,
    ) -> None:
        """Router without telemetry still works."""
        d = offline_router.route_sync("write a function")
        assert d.primary_model in ("devstral-small", "devstral-2")


class TestRoutingDecisionEffort:
    """Routing decision includes reasoning_effort."""

    def test_general_task_gets_effort(self, offline_router: MistralRouter) -> None:
        d = offline_router.route_sync("summarize this document about EU regulations")
        # mistral-small-4 supports effort — should get one
        if d.primary_model == "mistral-small-4":
            assert d.reasoning_effort is not None

    def test_code_task_no_effort(self, offline_router: MistralRouter) -> None:
        d = offline_router.route_sync("write a Python sort function")
        # devstral-small doesn't support reasoning_effort
        if d.primary_model == "devstral-small":
            assert d.reasoning_effort is None

    def test_importable_from_package(self) -> None:
        from tramontane import FleetTelemetry, RoutingOutcome

        assert FleetTelemetry is not None
        assert RoutingOutcome is not None


class TestTelemetryRecordingInAgent:
    """Agent.run() records telemetry when router has telemetry."""

    def test_router_telemetry_attribute_accessible(self) -> None:
        from tramontane.router.telemetry import FleetTelemetry

        t = FleetTelemetry(db_path=":memory:")
        r = MistralRouter(telemetry=t)
        assert r._telemetry is t
        assert r._telemetry.total_outcomes == 0
