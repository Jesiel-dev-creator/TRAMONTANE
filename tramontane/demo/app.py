"""TRAMONTANE — Mistral-Native Agent Orchestration Demo.

Two tabs: Router Explorer (offline) + Live Agent (real Mistral API).
IP-based rate limiting + global daily budget cap on live calls.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Any

import gradio as gr
import pandas as pd

from tramontane.core.agent import Agent
from tramontane.router.classifier import ClassificationMode, TaskClassifier
from tramontane.router.models import MISTRAL_MODELS
from tramontane.router.router import MistralRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HAS_API_KEY = bool(os.environ.get("MISTRAL_API_KEY"))

offline_classifier = TaskClassifier(mode=ClassificationMode.OFFLINE)
offline_router = MistralRouter(classifier=offline_classifier)

if HAS_API_KEY:
    online_router = MistralRouter()
else:
    online_router = offline_router

AGENT_PRESETS: dict[str, dict[str, str]] = {
    "Code Reviewer": {
        "role": "Senior Code Reviewer",
        "goal": "Review code for bugs, security issues, and best practices",
        "backstory": "10 years Python experience, security auditing specialist",
    },
    "Research Analyst": {
        "role": "Research Analyst",
        "goal": "Analyze topics and provide clear, structured insights",
        "backstory": "Expert researcher in technology and business analysis",
    },
    "Writing Assistant": {
        "role": "Writing Assistant",
        "goal": "Help write clear, engaging, professional content",
        "backstory": "Experienced editor fluent in English and French",
    },
    "General Assistant": {
        "role": "Helpful Assistant",
        "goal": "Answer questions accurately and helpfully",
        "backstory": "Knowledgeable assistant with broad expertise",
    },
}

EXAMPLES: list[list[Any]] = [
    ["Write a Python function to parse JSON and handle errors", 0.05, "en"],
    ["Analyze the impact of the EU AI Act on French SMEs", 0.10, "fr"],
    ["Search for the latest Mistral AI news", 0.02, "en"],
    ["List all EU member states", 0.005, "en"],
    ["Build a full-stack Next.js app with Supabase auth", 0.20, "en"],
    [
        "R\u00e9dige un email de prospection en fran\u00e7ais pour une PME",
        0.05,
        "fr",
    ],
]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_ip_calls: dict[str, list[float]] = defaultdict(list)
MAX_CALLS_PER_IP_PER_DAY = 5

_daily_spend: dict[str, object] = {"eur": 0.0, "date": ""}
MAX_DAILY_SPEND_EUR = 1.00


def check_rate_limit(request: gr.Request | None) -> tuple[bool, str]:
    """Check if this request is allowed."""
    ip = request.client.host if request and request.client else "unknown"
    today = time.strftime("%Y-%m-%d")

    if _daily_spend["date"] != today:
        _daily_spend["eur"] = 0.0
        _daily_spend["date"] = today
        _ip_calls.clear()
        logger.info("Daily rate limit counters reset")

    if float(str(_daily_spend["eur"])) >= MAX_DAILY_SPEND_EUR:
        return False, (
            "Daily demo budget reached. Try again tomorrow "
            "or install locally: `pip install tramontane`"
        )

    calls_today = _ip_calls.get(ip, [])
    if len(calls_today) >= MAX_CALLS_PER_IP_PER_DAY:
        return False, (
            f"You've used your {MAX_CALLS_PER_IP_PER_DAY} free calls today. "
            "Try again tomorrow or install locally: `pip install tramontane`"
        )

    _ip_calls[ip].append(time.time())
    return True, "ok"


def record_spend(cost_eur: float) -> None:
    """Record actual API spend against the global daily cap."""
    _daily_spend["eur"] = float(str(_daily_spend["eur"])) + cost_eur


# ---------------------------------------------------------------------------
# Fleet table
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
# Tab 1: Router Explorer (offline)
# ---------------------------------------------------------------------------


def route_task(
    task: str, budget_eur: float, locale: str,
) -> tuple[str, pd.DataFrame]:
    """Route a task offline and return explanation markdown + fleet table."""
    if not task.strip():
        return (
            "*Enter a task above and click 'Route Task' to see the routing decision.*",
            get_fleet_df(),
        )

    decision = offline_router.route_sync(
        task.strip(), budget=budget_eur, locale=locale,
    )
    explanation = offline_router.explain(decision)

    downgrade_note = ""
    if decision.downgrade_applied:
        downgrade_note = (
            f"\n**Budget constraint applied** \u2014 "
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
| **Function Calling** | `{decision.function_calling_model}` |
| **Reasoning Model** | `{decision.reasoning_model}` |
| **Budget Constrained** | `{decision.budget_constrained}` |

### Why this model?

> {explanation}
{downgrade_note}
---
*Router running in OFFLINE mode \u00b7 No API key required*
"""
    return md.strip(), get_fleet_df()


# ---------------------------------------------------------------------------
# Tab 2: Live Agent (real API)
# ---------------------------------------------------------------------------


async def run_live_agent(
    message: str,
    history: list[dict[str, str]],
    agent_type: str,
    budget: float,
    request: gr.Request,
) -> Any:
    """Run a real Mistral API call through Tramontane with rate limiting."""
    if not HAS_API_KEY:
        yield "Live mode unavailable \u2014 no API key configured on this Space."
        return

    allowed, limit_msg = check_rate_limit(request)
    if not allowed:
        yield limit_msg
        return

    preset = AGENT_PRESETS[agent_type]
    agent = Agent(
        role=preset["role"],
        goal=preset["goal"],
        backstory=preset["backstory"],
        model="auto",
        budget_eur=budget,
    )

    try:
        result = await agent.run(message, router=online_router)
        record_spend(result.cost_eur)

        ip = request.client.host if request and request.client else "unknown"
        calls_used = len(_ip_calls.get(ip, []))
        calls_remaining = MAX_CALLS_PER_IP_PER_DAY - calls_used

        metadata = (
            f"\n\n---\n"
            f"**Model:** `{result.model_used}` \u00b7 "
            f"**Cost:** \u20ac{result.cost_eur:.4f} \u00b7 "
            f"**Tokens:** {result.input_tokens} in \u2192 "
            f"{result.output_tokens} out \u00b7 "
            f"**{result.duration_seconds:.1f}s** \u00b7 "
            f"**Calls remaining:** {calls_remaining}/"
            f"{MAX_CALLS_PER_IP_PER_DAY}"
        )

        yield result.output + metadata

    except Exception as exc:
        logger.error("Live agent error: %s", exc, exc_info=True)
        yield f"Error: {exc}"


# ---------------------------------------------------------------------------
# CSS — EU Premium Design System
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');

.gradio-container {
    max-width: 1100px !important;
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #00D4EE 0%, #00B4CC 100%) !important;
    border: none !important;
    color: #020408 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
textarea, input[type="text"] {
    background: #060C14 !important;
    border: 1px solid #1C2E42 !important;
    color: #DCE9F5 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #00D4EE !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 238, 0.12) !important;
}
code, pre {
    font-family: 'Space Mono', monospace !important;
    background: #0D1B2A !important;
    border: 1px solid #1C2E42 !important;
    border-radius: 6px !important;
}
th {
    background: #0D1B2A !important;
    color: #00D4EE !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
}
.tab-nav button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}
.tab-nav button.selected {
    color: #00D4EE !important;
    border-bottom: 3px solid #00D4EE !important;
}
label {
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 12px !important;
    letter-spacing: 0.8px !important;
}
footer { display: none !important; }
.chatbot { overflow-y: auto !important; scrollbar-width: none !important; }
.chatbot::-webkit-scrollbar { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TRAMONTANE \u2014 Mistral-Native Agent Orchestration",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ).set(
        body_background_fill="#020408",
        body_text_color="#DCE9F5",
        block_background_fill="#0D1B2A",
        block_border_color="#1C2E42",
        input_background_fill="#060C14",
        button_primary_background_fill="#00D4EE",
        button_primary_text_color="#020408",
    ),
    css=CSS,
) as demo:

    gr.Markdown(
        "# TRAMONTANE\n"
        "**Mistral-native agent orchestration \u00b7 "
        "Built in Orl\u00e9ans, France**\n\n"
        "Smart routing across 10 Mistral models \u00b7 "
        "GDPR built-in \u00b7 EUR budget control \u00b7 MIT License\n\n"
        "`pip install tramontane` \u00b7 "
        "[GitHub](https://github.com/Jesiel-dev-creator/TRAMONTANE) \u00b7 "
        "[PyPI](https://pypi.org/project/tramontane/)\n\n---"
    )

    with gr.Tabs():

        # ── TAB 1: Router Explorer ──
        with gr.TabItem("Router Explorer"):
            gr.Markdown(
                "**See which Mistral model gets selected for your task.**\n"
                "The router classifies your prompt, then picks the optimal "
                "model \u2014 balancing capability, cost, and budget. "
                "No API key needed."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    task_input = gr.Textbox(
                        label="DESCRIBE YOUR TASK",
                        placeholder=(
                            "e.g., Write a Python function to parse JSON "
                            "and handle errors..."
                        ),
                        lines=4,
                        max_lines=8,
                    )
                    with gr.Row():
                        budget_slider = gr.Slider(
                            minimum=0.001,
                            maximum=0.50,
                            value=0.05,
                            step=0.001,
                            label="BUDGET CEILING (\u20ac)",
                        )
                        locale_drop = gr.Dropdown(
                            choices=["en", "fr", "de", "es", "it", "pt"],
                            value="en",
                            label="LOCALE",
                        )
                    route_btn = gr.Button(
                        "Route Task", variant="primary", size="lg",
                    )

                with gr.Column(scale=3):
                    result_md = gr.Markdown(
                        value=(
                            "*Enter a task above and click "
                            "'Route Task' to see the routing decision.*"
                        ),
                    )

            gr.Markdown("### Try These Examples")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[task_input, budget_slider, locale_drop],
                label=None,
            )

        # ── TAB 2: Live Agent ──
        with gr.TabItem("Live Agent"):
            if HAS_API_KEY:
                gr.Markdown(
                    "**Talk to a real Mistral agent through Tramontane.**\n"
                    "The router automatically selects the optimal model.\n\n"
                    f"*{MAX_CALLS_PER_IP_PER_DAY} free calls per day \u00b7 "
                    "Budget capped per call \u00b7 Powered by Mistral AI*"
                )

                with gr.Row():
                    agent_type = gr.Dropdown(
                        choices=list(AGENT_PRESETS.keys()),
                        value="General Assistant",
                        label="AGENT TYPE",
                        scale=2,
                    )
                    live_budget = gr.Slider(
                        minimum=0.001,
                        maximum=0.05,
                        value=0.01,
                        step=0.001,
                        label="MAX COST PER CALL (\u20ac)",
                        scale=1,
                    )

                gr.ChatInterface(
                    fn=run_live_agent,
                    type="messages",
                    additional_inputs=[agent_type, live_budget],
                    chatbot=gr.Chatbot(
                        height=480,
                        type="messages",
                        placeholder="Type a message to talk to the agent...",
                    ),
                )
            else:
                gr.Markdown(
                    "### Live Agent Unavailable\n\n"
                    "This Space is running without an API key.\n\n"
                    "**To try the live agent locally:**\n\n"
                    "```bash\npip install tramontane\n"
                    "export MISTRAL_API_KEY=your_key\n"
                    "python examples/quickstart.py\n```\n\n"
                    "**Try the Router Explorer tab** to see how "
                    "intelligent model routing works!"
                )

    with gr.Accordion("Mistral Model Fleet", open=False):
        fleet_table = gr.Dataframe(
            value=get_fleet_df(), interactive=False, wrap=True,
        )

    gr.Markdown(
        "---\n"
        "**TRAMONTANE v0.1.3** \u00b7 MIT License \u00b7 "
        "[GitHub](https://github.com/Jesiel-dev-creator/TRAMONTANE) \u00b7 "
        "[PyPI](https://pypi.org/project/tramontane/) \u00b7 "
        "Built in Orl\u00e9ans, France by **Bleucommerce SAS** \u00b7 "
        "Powered by **Mistral AI**"
    )

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

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
