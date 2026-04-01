"""Mistral model router — the core differentiator of Tramontane.

Routes every prompt to the optimal Mistral model based on task type,
complexity, budget, locale, and local-mode constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from tramontane.core.exceptions import BudgetExceededError
from tramontane.router.classifier import (
    ClassificationResult,
    TaskClassifier,
)
from tramontane.router.models import (
    MISTRAL_MODELS,
    MistralModel,
    get_model,
)

logger = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = Path(__file__).parent / "rules.yaml"

# Locale codes that benefit from strong multilingual models
_MULTILINGUAL_LOCALES: set[str] = {"fr", "de", "es", "it", "pt"}

# Minimum quality floors per task type.
# Budget downgrade NEVER goes below these — failing loudly is better
# than silently serving garbage from an incapable model.
_QUALITY_FLOORS: dict[str, str] = {
    "reasoning": "magistral-small",
    "code": "devstral-small",
    "vision": "pixtral-large",
    "research": "mistral-small",
    "general": "mistral-small",
    "bulk": "ministral-7b",
    "classification": "ministral-3b",
    "voice": "ministral-7b",
}


class RoutingDecision(BaseModel):
    """The result of routing a prompt to a Mistral model."""

    primary_model: str
    function_calling_model: str
    reasoning_model: str
    classification: ClassificationResult
    budget_constrained: bool
    local_mode: bool
    estimated_cost_eur: float
    downgrade_applied: bool
    downgrade_reason: str | None = None


class MistralRouter:
    """Routes prompts to the optimal Mistral model.

    Loads routing rules from YAML and uses the TaskClassifier to
    determine task type and complexity before applying the decision tree.
    """

    def __init__(
        self,
        rules_path: str | None = None,
        local_mode: bool = False,
        classifier: TaskClassifier | None = None,
    ) -> None:
        path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        self._rules: dict[str, Any] = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
        self._local_mode = local_mode
        self._classifier = classifier or TaskClassifier()

    # -----------------------------------------------------------------
    # Main routing
    # -----------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        agent_budget_eur: float | None = None,
        locale: str = "en",
        context: str | None = None,
        force_model: str | None = None,
    ) -> RoutingDecision:
        """Route a prompt to the optimal Mistral model."""
        classification = await self._classifier.classify(prompt, context)

        downgrade_applied = False
        downgrade_reason: str | None = None
        budget_constrained = False

        # -- Force model shortcut ------------------------------------------
        if force_model:
            model_info = get_model(force_model)
            cost = self._estimate_cost(
                model_info, classification.estimated_output_tokens
            )
            return RoutingDecision(
                primary_model=force_model,
                function_calling_model=self._resolve_fc_model(classification),
                reasoning_model=self._resolve_reasoning_model(),
                classification=classification,
                budget_constrained=False,
                local_mode=self._local_mode,
                estimated_cost_eur=cost,
                downgrade_applied=False,
            )

        # -- Decision tree (from CLAUDE.md) --------------------------------
        primary = self._decide_primary(classification)

        # -- Budget constraint ---------------------------------------------
        if agent_budget_eur is not None:
            model_info = get_model(primary)
            cost = self._estimate_cost(
                model_info, classification.estimated_output_tokens
            )
            if cost > agent_budget_eur:
                budget_constrained = True
                primary, downgrade_applied, downgrade_reason = (
                    self._apply_budget_downgrade(
                        budget_eur=agent_budget_eur,
                        est_output_tokens=classification.estimated_output_tokens,
                        task_type=classification.task_type,
                        needs_reasoning=classification.needs_reasoning,
                    )
                )

        # -- Locale preference ---------------------------------------------
        if (
            locale in _MULTILINGUAL_LOCALES
            and classification.task_type != "code"
        ):
            strong: list[str] = (
                self._rules.get("locale_preference", {})
                .get("strong_multilingual", [])
            )
            if primary not in strong and strong:
                primary = strong[0]

        # -- Local mode (Ollama mapping) -----------------------------------
        if self._local_mode:
            primary = self._map_to_local(primary)

        # -- Function-calling + reasoning models ---------------------------
        fc_model = self._resolve_fc_model(classification)
        reasoning_model = self._resolve_reasoning_model()

        if self._local_mode:
            fc_model = self._map_to_local(fc_model)
            reasoning_model = self._map_to_local(reasoning_model)

        model_info = get_model(primary)
        estimated_cost = self._estimate_cost(
            model_info, classification.estimated_output_tokens
        )

        return RoutingDecision(
            primary_model=primary,
            function_calling_model=fc_model,
            reasoning_model=reasoning_model,
            classification=classification,
            budget_constrained=budget_constrained,
            local_mode=self._local_mode,
            estimated_cost_eur=estimated_cost,
            downgrade_applied=downgrade_applied,
            downgrade_reason=downgrade_reason,
        )

    def route_sync(
        self,
        prompt: str,
        budget: float | None = None,
        locale: str = "en",
        context: str | None = None,
        force_model: str | None = None,
    ) -> RoutingDecision:
        """Synchronous wrapper for route(). Do not call from async context."""
        from tramontane.core._sync import run_sync

        return run_sync(
            self.route(
                prompt,
                agent_budget_eur=budget,
                locale=locale,
                context=context,
                force_model=force_model,
            )
        )

    # -----------------------------------------------------------------
    # Human-readable explanation
    # -----------------------------------------------------------------

    @staticmethod
    def explain(decision: RoutingDecision) -> str:
        """Return a human-readable explanation of a routing decision."""
        parts = [
            f"Routed to {decision.primary_model}",
            f"({decision.classification.task_type} task",
            f"complexity {decision.classification.complexity}",
            f"EUR {decision.estimated_cost_eur:.4f} est.)",
        ]
        if decision.downgrade_applied:
            parts.append(f"[downgraded: {decision.downgrade_reason}]")
        if decision.local_mode:
            parts.append("[local/Ollama]")
        return " ".join(parts)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _decide_primary(self, cls_result: ClassificationResult) -> str:
        """Apply the CLAUDE.md decision tree to pick the primary model."""
        routing = self._rules.get("routing", {})

        if cls_result.has_vision:
            return str(routing.get("vision", {}).get("default", "pixtral-large"))

        if cls_result.task_type == "code" or cls_result.has_code:
            code_rules = routing.get("code", {})
            if cls_result.complexity >= 4:
                return str(code_rules.get("complex", "devstral-2"))
            return str(code_rules.get("default", "devstral-small"))

        if cls_result.needs_reasoning:
            reasoning_rules = routing.get("reasoning", {})
            if cls_result.complexity >= 4:
                return str(reasoning_rules.get("deep", "magistral-medium"))
            return str(reasoning_rules.get("default", "magistral-small"))

        if cls_result.task_type == "bulk" or cls_result.complexity == 1:
            return str(routing.get("bulk", {}).get("default", "ministral-7b"))

        if (
            cls_result.task_type == "research"
            and cls_result.complexity >= 3
        ):
            return str(
                routing.get("research", {}).get("frontier", "mistral-large")
            )

        if cls_result.task_type == "classification":
            return str(
                routing.get("classification", {}).get("default", "ministral-3b")
            )

        if cls_result.task_type == "voice":
            return str(
                routing.get("voice", {}).get("default", "ministral-7b")
            )

        return str(routing.get("general", {}).get("default", "mistral-small"))

    def _apply_budget_downgrade(
        self,
        budget_eur: float,
        est_output_tokens: int,
        task_type: str,
        needs_reasoning: bool,
    ) -> tuple[str, bool, str]:
        """Downgrade model for budget, but never below quality floor.

        Uses _QUALITY_FLOORS to prevent serving an incapable model.
        Raises BudgetExceededError if budget is insufficient even for
        the minimum quality model.
        """
        # Determine the floor for this task
        effective_type = task_type
        if needs_reasoning and effective_type not in ("code", "vision"):
            effective_type = "reasoning"
        floor_alias = _QUALITY_FLOORS.get(effective_type, "mistral-small")
        floor_info = MISTRAL_MODELS.get(floor_alias)
        floor_tier = floor_info.tier if floor_info else 0

        # Check if even the floor model fits the budget
        if floor_info:
            floor_cost = self._estimate_cost(floor_info, est_output_tokens)
            if floor_cost > budget_eur:
                raise BudgetExceededError(
                    budget_eur=budget_eur,
                    spent_eur=0.0,
                    pipeline_name=(
                        f"budget EUR {budget_eur:.4f} insufficient for "
                        f"minimum quality model '{floor_alias}' "
                        f"(estimated EUR {floor_cost:.4f}, "
                        f"task type '{task_type}' requires tier >= {floor_tier})"
                    ),
                )

        # Find cheapest model that meets BOTH budget AND quality floor
        candidates: list[tuple[float, str]] = []
        for alias, model in MISTRAL_MODELS.items():
            if not model.available or model.tier < floor_tier:
                continue
            cost = self._estimate_cost(model, est_output_tokens)
            if cost <= budget_eur:
                candidates.append((cost, alias))

        if not candidates:
            raise BudgetExceededError(
                budget_eur=budget_eur,
                spent_eur=0.0,
                pipeline_name=(
                    f"no model with tier >= {floor_tier} fits "
                    f"budget EUR {budget_eur:.4f} for task type '{task_type}'"
                ),
            )

        candidates.sort()
        chosen = candidates[0][1]
        return (
            chosen,
            True,
            f"budget: EUR {budget_eur:.4f} limit (floor: {floor_alias})",
        )

    def _resolve_fc_model(self, cls_result: ClassificationResult) -> str:
        """Resolve the function-calling model."""
        fc_rules = self._rules.get("function_calling", {})
        if cls_result.needs_reasoning:
            return str(fc_rules.get("reasoning_required", "magistral-small"))
        return str(fc_rules.get("default", "ministral-7b"))

    def _resolve_reasoning_model(self) -> str:
        """Resolve the reasoning model."""
        return "magistral-small"

    def _map_to_local(self, alias: str) -> str:
        """Map a model alias to its Ollama equivalent."""
        model = MISTRAL_MODELS.get(alias)
        if model and model.local_ollama:
            return alias  # keep alias, Ollama name is in model registry
        # Fallback chain for models without Ollama equivalent
        for fallback in ("mistral-small", "ministral-7b"):
            fb = MISTRAL_MODELS.get(fallback)
            if fb and fb.local_ollama:
                return fallback
        return alias

    @staticmethod
    def _estimate_cost(model_info: MistralModel, est_output_tokens: int) -> float:
        """Estimate EUR cost for a given model and output token count."""
        return (est_output_tokens / 1_000_000) * model_info.cost_per_1m_output_eur
