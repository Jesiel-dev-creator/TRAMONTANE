"""TRAMONTANE — Mistral Router Demo (HuggingFace Space).

Demonstrates the intelligent model router in OFFLINE mode.
No API key needed. Zero cold-start dependencies beyond tramontane + gradio.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from tramontane.router.classifier import ClassificationMode, TaskClassifier
from tramontane.router.models import MISTRAL_MODELS
from tramontane.router.router import MistralRouter

# ---------------------------------------------------------------------------
# Initialise router in OFFLINE mode (no API key needed on HF Space)
# ---------------------------------------------------------------------------

classifier = TaskClassifier(mode=ClassificationMode.OFFLINE)
router = MistralRouter(classifier=classifier)


# ---------------------------------------------------------------------------
# Model fleet table
# ---------------------------------------------------------------------------


def get_fleet_df() -> pd.DataFrame:
    """Build the Mistral model fleet as a sorted DataFrame."""
    rows = []
    for alias, m in MISTRAL_MODELS.items():
        rows.append({
            "Model": alias,
            "Tier": m.tier,
            "Best For": ", ".join(m.strengths[:2]),
            "\u20ac/1M in": f"\u20ac{m.cost_per_1m_input_eur:.2f}",
            "\u20ac/1M out": f"\u20ac{m.cost_per_1m_output_eur:.2f}",
            "Local": "\u2713 ollama" if m.local_ollama else "\u2014",
        })
    return pd.DataFrame(rows).sort_values("Tier")


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------


def route_task(
    task: str, budget_eur: float, locale: str,
) -> tuple[str, pd.DataFrame]:
    """Route a task and return formatted Markdown + fleet table."""
    if not task.strip():
        return "Enter a task description above.", get_fleet_df()

    decision = router.route_sync(
        task.strip(), budget=budget_eur, locale=locale,
    )
    explanation = router.explain(decision)

    downgrade_note = ""
    if decision.downgrade_applied:
        downgrade_note = (
            f"\n\u26a0\ufe0f **Budget constraint applied** \u2014 "
            f"{decision.downgrade_reason}\n"
        )

    md = f"""
## Routing Decision

| Field | Value |
|-------|-------|
| **Selected Model** | `{decision.primary_model}` |
| **Estimated Cost** | `\u20ac{decision.estimated_cost_eur:.4f}` |
| **Task Type** | `{decision.classification.task_type}` |
| **Complexity** | `{decision.classification.complexity}/5` |
| **Needs Reasoning** | `{decision.classification.needs_reasoning}` |
| **Has Code** | `{decision.classification.has_code}` |
| **Language** | `{decision.classification.language}` |
| **Function Calling Model** | `{decision.function_calling_model}` |
| **Reasoning Model** | `{decision.reasoning_model}` |
| **Budget Constrained** | `{decision.budget_constrained}` |
| **Downgrade Applied** | `{decision.downgrade_applied}` |

### Explanation

> {explanation}
{downgrade_note}
---
*Router running in OFFLINE mode \u00b7 No API key required*
"""
    return md.strip(), get_fleet_df()


# ---------------------------------------------------------------------------
# Example tasks
# ---------------------------------------------------------------------------

EXAMPLES = [
    ["Write a Python function to parse JSON and handle errors", 0.05, "en"],
    ["Analyze the impact of the EU AI Act on French SMEs", 0.10, "fr"],
    ["Search for the latest Mistral AI news", 0.02, "en"],
    ["List all EU member states", 0.005, "en"],
    ["Build a full-stack Next.js app with Supabase auth", 0.20, "en"],
    [
        "R\u00e9dige un email de prospection en fran\u00e7ais pour une PME",
        0.05, "fr",
    ],
]


# ---------------------------------------------------------------------------
# EU Premium CSS
# ---------------------------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1000px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gr-button-primary {
    background: #00D4EE !important;
    border: none !important;
    color: #020408 !important;
    font-weight: 700 !important;
}
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TRAMONTANE \u2014 Mistral Router Demo",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ).set(
        body_background_fill="#020408",
        body_text_color="#DCE9F5",
        block_background_fill="#0D1B2A",
        block_border_color="#1C2E42",
        input_background_fill="#060C14",
    ),
    css=CSS,
) as demo:

    gr.Markdown("""
# TRAMONTANE
### Mistral-native agent orchestration \u00b7 Model Router Demo

See which Mistral model gets automatically selected for your task.
The router picks the **optimal model** \u2014 balancing capability, cost, and your budget.

---
""")

    with gr.Row():
        with gr.Column(scale=2):
            task_input = gr.Textbox(
                label="Describe your task",
                placeholder="Write a Python function to parse JSON...",
                lines=3,
            )
            with gr.Row():
                budget_slider = gr.Slider(
                    minimum=0.001,
                    maximum=0.50,
                    value=0.05,
                    step=0.001,
                    label="Budget ceiling (\u20ac)",
                )
                locale_drop = gr.Dropdown(
                    choices=["en", "fr", "de", "es", "it", "pt"],
                    value="en",
                    label="Locale",
                )
            route_btn = gr.Button("Route Task", variant="primary")

        with gr.Column(scale=3):
            result_md = gr.Markdown(
                value="*Enter a task above to see the routing decision.*",
            )

    gr.Markdown("### Try These Examples")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[task_input, budget_slider, locale_drop],
        label=None,
    )

    gr.Markdown("### Mistral Model Fleet")
    fleet_table = gr.Dataframe(
        value=get_fleet_df(),
        interactive=False,
        wrap=True,
    )

    gr.Markdown("""
---
**TRAMONTANE** \u00b7 MIT License \u00b7
[GitHub](https://github.com/Jesiel-dev-creator/TRAMONTANE) \u00b7
Built in Orl\u00e9ans, France by **Bleucommerce SAS**
""")

    route_btn.click(
        fn=route_task,
        inputs=[task_input, budget_slider, locale_drop],
        outputs=[result_md, fleet_table],
    )
    task_input.submit(
        fn=route_task,
        inputs=[task_input, budget_slider, locale_drop],
        outputs=[result_md, fleet_table],
    )

if __name__ == "__main__":
    demo.launch()
