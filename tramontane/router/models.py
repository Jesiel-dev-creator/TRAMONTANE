"""Mistral model fleet registry.

Complete registry of all Mistral models available for routing,
with cost data in EUR and helper functions for budget-aware selection.
"""

from __future__ import annotations

from pydantic import BaseModel

from tramontane.core.exceptions import ModelNotAvailableError


class MistralModel(BaseModel):
    """Describes a single Mistral model available for routing."""

    api_id: str
    hf_id: str | None = None
    tier: int  # 0-4, lower = cheaper
    cost_per_1m_input_eur: float
    cost_per_1m_output_eur: float
    context_window: int
    strengths: list[str]
    local_ollama: str | None = None
    license: str
    hf_downloads: int = 0
    max_output_tokens: int = 8192
    modality: str = "text"  # text | audio | vision
    available: bool = True


MISTRAL_MODELS: dict[str, MistralModel] = {
    "ministral-3b": MistralModel(
        api_id="ministral-3b-latest",
        tier=0,
        cost_per_1m_input_eur=0.04,
        cost_per_1m_output_eur=0.04,
        context_window=128_000,
        max_output_tokens=8192,
        strengths=["classification", "pii-detection", "triage"],
        local_ollama="ministral:3b",
        license="MRL",
    ),
    "ministral-7b": MistralModel(
        api_id="ministral-8b-latest",
        tier=1,
        cost_per_1m_input_eur=0.10,
        cost_per_1m_output_eur=0.10,
        context_window=128_000,
        max_output_tokens=8192,
        strengths=["bulk", "extraction", "tool-calls", "function-calling"],
        local_ollama="ministral:8b",
        license="MRL",
    ),
    "mistral-small": MistralModel(
        api_id="mistral-small-latest",
        tier=2,
        cost_per_1m_input_eur=0.10,
        cost_per_1m_output_eur=0.30,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["general", "multilingual", "default"],
        local_ollama="mistral-small",
        license="MRL",
    ),
    "devstral-small": MistralModel(
        api_id="devstral-small-latest",
        tier=2,
        cost_per_1m_input_eur=0.10,
        cost_per_1m_output_eur=0.30,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["code", "swe", "all-code-tasks"],
        local_ollama="devstral:small",
        license="MRL",
    ),
    "magistral-small": MistralModel(
        api_id="magistral-small-latest",
        tier=3,
        cost_per_1m_input_eur=0.50,
        cost_per_1m_output_eur=1.50,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["reasoning", "cot", "planning"],
        local_ollama="magistral:small",
        license="MRL",
    ),
    "magistral-medium": MistralModel(
        api_id="magistral-medium-latest",
        tier=3,
        cost_per_1m_input_eur=2.00,
        cost_per_1m_output_eur=5.00,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["deep-reasoning", "complex-cot"],
        license="MRL",
    ),
    "devstral-2": MistralModel(
        api_id="devstral-latest",
        tier=4,
        cost_per_1m_input_eur=0.50,
        cost_per_1m_output_eur=1.50,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["complex-swe", "monorepo", "large-codebase"],
        license="MRL",
    ),
    "pixtral-large": MistralModel(
        api_id="pixtral-large-latest",
        tier=4,
        cost_per_1m_input_eur=2.00,
        cost_per_1m_output_eur=6.00,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["vision", "multimodal", "ocr"],
        license="MRL",
        modality="vision",
    ),
    "mistral-large": MistralModel(
        api_id="mistral-large-latest",
        tier=4,
        cost_per_1m_input_eur=2.00,
        cost_per_1m_output_eur=6.00,
        context_window=128_000,
        max_output_tokens=32768,
        strengths=["frontier", "synthesis", "research"],
        license="MRL",
    ),
    "voxtral-mini": MistralModel(
        api_id="voxtral-mini-latest",
        tier=1,
        cost_per_1m_input_eur=0.04,
        cost_per_1m_output_eur=0.04,
        context_window=32_000,
        max_output_tokens=8192,
        strengths=["voice", "transcription", "speech-input"],
        license="MRL",
        modality="audio",
    ),
}


def get_model(alias: str) -> MistralModel:
    """Look up a model by its Tramontane alias.

    Raises ModelNotAvailableError if the alias is unknown or the model
    is marked as unavailable.
    """
    model = MISTRAL_MODELS.get(alias)
    if model is None:
        raise ModelNotAvailableError(model=alias, reason="unknown alias")
    if not model.available:
        raise ModelNotAvailableError(model=alias, reason="model is currently unavailable")
    return model


def models_by_tier(tier: int) -> list[MistralModel]:
    """Return all models in a given tier, sorted by input cost ascending."""
    return sorted(
        (m for m in MISTRAL_MODELS.values() if m.tier == tier),
        key=lambda m: m.cost_per_1m_input_eur,
    )


def cheapest_model_for_budget(
    budget_eur: float,
    output_tokens_estimate: int,
) -> MistralModel | None:
    """Find the cheapest available model that fits within the given budget.

    Uses output_tokens_estimate to calculate expected cost.
    Returns None if no model fits.
    """
    candidates: list[tuple[float, MistralModel]] = []
    for model in MISTRAL_MODELS.values():
        if not model.available:
            continue
        estimated_cost = (output_tokens_estimate / 1_000_000) * model.cost_per_1m_output_eur
        if estimated_cost <= budget_eur:
            candidates.append((estimated_cost, model))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]
