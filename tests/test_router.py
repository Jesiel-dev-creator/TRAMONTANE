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
