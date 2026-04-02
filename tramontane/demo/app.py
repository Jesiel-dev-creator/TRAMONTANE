"""TRAMONTANE v0.2.2 — The Evolution Release Demo.

Four tabs: Router Explorer | Fleet Intelligence | Live Agent | Benchmark
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
from tramontane.core.profiles import PROFILE_CONFIGS, FleetProfile
from tramontane.core.simulate import simulate_agent
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
    ["Build a full-stack Next.js app with Supabase auth", 0.20, "en"],
    ["Create a design system with warm colors for a bakery", 0.05, "en"],
    ["Classify this email as spam or not spam", 0.005, "en"],
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

    if float(str(_daily_spend["eur"])) >= MAX_DAILY_SPEND_EUR:
        return False, "Daily demo budget reached. Install locally: `pip install tramontane`"

    calls_today = _ip_calls.get(ip, [])
    if len(calls_today) >= MAX_CALLS_PER_IP_PER_DAY:
        return False, f"Used {MAX_CALLS_PER_IP_PER_DAY} free calls. Try tomorrow."

    _ip_calls[ip].append(time.time())
    return True, "ok"


def record_spend(cost_eur: float) -> None:
    """Record actual API spend."""
    _daily_spend["eur"] = float(str(_daily_spend["eur"])) + cost_eur


# ---------------------------------------------------------------------------
# Fleet table (updated for v0.2.2)
# ---------------------------------------------------------------------------


def get_fleet_df() -> pd.DataFrame:
    """Build the Mistral model fleet as a DataFrame."""
    rows = []
    for alias, m in MISTRAL_MODELS.items():
        rows.append({
            "Model": alias,
            "Tier": m.tier,
            "Best For": ", ".join(m.strengths[:2]),
            "\u20ac/1M in": f"\u20ac{m.cost_per_1m_input_eur:.2f}",
            "\u20ac/1M out": f"\u20ac{m.cost_per_1m_output_eur:.2f}",
            "Context": f"{m.context_window // 1000}K",
            "Reasoning": "\u2713" if m.supports_reasoning_effort else "\u2014",
            "Vision": "\u2713" if m.supports_vision else "\u2014",
        })
    return pd.DataFrame(rows).sort_values("Tier")


# ---------------------------------------------------------------------------
# Tab 1: Router Explorer (offline)
# ---------------------------------------------------------------------------


def route_task(
    task: str, budget_eur: float, locale: str,
) -> tuple[str, pd.DataFrame]:
    """Route a task offline and return explanation + fleet table."""
    if not task.strip():
        return "*Enter a task and click 'Route Task'.*", get_fleet_df()

    decision = offline_router.route_sync(
        task.strip(), budget=budget_eur, locale=locale,
    )

    downgrade_note = ""
    if decision.downgrade_applied:
        downgrade_note = (
            f"\n**Budget constraint** \u2014 {decision.downgrade_reason}\n"
        )

    effort_str = decision.reasoning_effort or "N/A"

    md = f"""
## Routing Decision

| Field | Value |
|-------|-------|
| **Selected Model** | `{decision.primary_model}` |
| **Reasoning Effort** | `{effort_str}` |
| **Estimated Cost** | `\u20ac{decision.estimated_cost_eur:.4f}` |
| **Task Type** | `{decision.classification.task_type}` |
| **Complexity** | `{decision.classification.complexity}/5` |
| **Needs Reasoning** | `{decision.classification.needs_reasoning}` |
| **Has Code** | `{decision.classification.has_code}` |
| **Language** | `{decision.classification.language}` |
| **Budget Constrained** | `{decision.budget_constrained}` |
{downgrade_note}
---
*v0.2.2 \u00b7 OFFLINE mode \u00b7 13 models \u00b7 No API key required*
"""
    return md.strip(), get_fleet_df()


# ---------------------------------------------------------------------------
# Tab 2: Fleet Intelligence (cost simulation + profiles)
# ---------------------------------------------------------------------------


def simulate_task(
    task: str, profile: str,
) -> str:
    """Simulate cost for a task with a fleet profile."""
    if not task.strip():
        return "*Enter a task to simulate.*"

    profile_enum = FleetProfile(profile.lower())
    config = PROFILE_CONFIGS[profile_enum]

    agent = Agent(
        role="Simulated Agent",
        goal="Process the task",
        backstory="General purpose agent",
        model=config.default_model,
        reasoning_effort=config.default_reasoning_effort,  # type: ignore[arg-type]
    )

    sim = simulate_agent(agent, task)

    md = f"""
## Cost Simulation

| Field | Value |
|-------|-------|
| **Profile** | `{profile}` |
| **Model** | `{sim.model_predicted}` |
| **Reasoning Effort** | `{sim.reasoning_effort or 'default'}` |
| **Est. Input Tokens** | `{sim.estimated_input_tokens}` |
| **Est. Output Tokens** | `{sim.estimated_output_tokens}` |
| **Est. Cost** | `\u20ac{sim.estimated_cost_eur:.6f}` |
| **Est. Time** | `{sim.estimated_time_s}s` |

### Profile Description
{config.description}

### What this means
- **BUDGET**: cheapest models, ~\u20ac0.001/call
- **BALANCED**: router decides everything (default)
- **QUALITY**: best models + deep reasoning, ~\u20ac0.005/call
- **UNIFIED**: mistral-small-4 for everything

---
*Simulation only \u2014 no API calls made*
"""
    return md.strip()


# ---------------------------------------------------------------------------
# Tab 3: Live Agent
# ---------------------------------------------------------------------------


async def run_live_agent(
    message: str,
    history: list[dict[str, str]],
    agent_type: str,
    budget: float,
    request: gr.Request,
) -> Any:
    """Run a real Mistral API call through Tramontane."""
    if not HAS_API_KEY:
        yield "Live mode unavailable \u2014 no API key configured."
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
        calls_remaining = MAX_CALLS_PER_IP_PER_DAY - len(_ip_calls.get(ip, []))

        metadata = (
            f"\n\n---\n"
            f"**Model:** `{result.model_used}` \u00b7 "
            f"**Cost:** \u20ac{result.cost_eur:.4f} \u00b7 "
            f"**Tokens:** {result.input_tokens}\u2192{result.output_tokens} \u00b7 "
            f"**{result.duration_seconds:.1f}s** \u00b7 "
            f"**Remaining:** {calls_remaining}/{MAX_CALLS_PER_IP_PER_DAY}"
        )

        yield result.output + metadata

    except Exception as exc:
        logger.error("Live agent error: %s", exc, exc_info=True)
        yield f"Error: {exc}"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;700&family=Space+Mono&display=swap');
.gradio-container { max-width: 1100px !important; font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #00D4EE 0%, #00B4CC 100%) !important;
    border: none !important; color: #020408 !important; font-weight: 700 !important;
}
textarea, input[type="text"] {
    background: #060C14 !important; border: 1px solid #1C2E42 !important;
    color: #DCE9F5 !important; border-radius: 10px !important;
}
code, pre { font-family: 'Space Mono', monospace !important; background: #0D1B2A !important; }
th { background: #0D1B2A !important; color: #00D4EE !important; font-weight: 700 !important; }
.tab-nav button { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
.tab-nav button.selected { color: #00D4EE !important; border-bottom: 3px solid #00D4EE !important; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TRAMONTANE v0.2.2 \u2014 The Evolution Release",
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
        "# TRAMONTANE v0.2.2\n"
        "**The only agent framework that gets smarter every time you use it.**\n\n"
        "13 Mistral models \u00b7 Smart routing \u00b7 "
        "Progressive reasoning \u00b7 Tool calling \u00b7 "
        "RAG \u00b7 GDPR native \u00b7 EUR budget control\n\n"
        "`pip install tramontane` \u00b7 "
        "[GitHub](https://github.com/Jesiel-dev-creator/TRAMONTANE) \u00b7 "
        "[PyPI](https://pypi.org/project/tramontane/) \u00b7 "
        "[Docs](https://github.com/Jesiel-dev-creator/TRAMONTANE/tree/main/docs)"
        "\n\n---"
    )

    with gr.Tabs():

        # Tab 1: Router Explorer
        with gr.TabItem("Router Explorer"):
            gr.Markdown(
                "**See which model gets selected for your task.**\n"
                "Offline classification \u2014 no API key needed."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    task_input = gr.Textbox(
                        label="TASK", lines=4,
                        placeholder="e.g., Write a Python sort function...",
                    )
                    with gr.Row():
                        budget_slider = gr.Slider(
                            0.001, 0.50, 0.05, step=0.001,
                            label="BUDGET (\u20ac)",
                        )
                        locale_drop = gr.Dropdown(
                            ["en", "fr", "de", "es", "it", "pt"],
                            value="en", label="LOCALE",
                        )
                    route_btn = gr.Button("Route Task", variant="primary")

                with gr.Column(scale=3):
                    result_md = gr.Markdown("*Enter a task and click 'Route Task'.*")

            gr.Examples(examples=EXAMPLES, inputs=[task_input, budget_slider, locale_drop])

        # Tab 2: Fleet Intelligence
        with gr.TabItem("Fleet Intelligence"):
            gr.Markdown(
                "**Simulate costs across fleet profiles without API calls.**\n"
                "Compare BUDGET vs QUALITY vs UNIFIED strategies."
            )
            with gr.Row():
                sim_task = gr.Textbox(
                    label="TASK TO SIMULATE", lines=3,
                    placeholder="e.g., Analyze the EU AI market...",
                )
                sim_profile = gr.Dropdown(
                    ["budget", "balanced", "quality", "unified"],
                    value="balanced", label="FLEET PROFILE",
                )
            sim_btn = gr.Button("Simulate Cost", variant="primary")
            sim_result = gr.Markdown("*Enter a task and select a profile.*")

            sim_btn.click(
                fn=simulate_task, inputs=[sim_task, sim_profile],
                outputs=[sim_result],
            )

        # Tab 3: Live Agent
        with gr.TabItem("Live Agent"):
            if HAS_API_KEY:
                gr.Markdown(
                    "**Talk to a real Mistral agent.**\n"
                    f"*{MAX_CALLS_PER_IP_PER_DAY} free calls/day*"
                )
                with gr.Row():
                    agent_type = gr.Dropdown(
                        list(AGENT_PRESETS.keys()), value="General Assistant",
                        label="AGENT TYPE", scale=2,
                    )
                    live_budget = gr.Slider(
                        0.001, 0.05, 0.01, step=0.001,
                        label="MAX COST (\u20ac)", scale=1,
                    )
                gr.ChatInterface(
                    fn=run_live_agent, type="messages",
                    additional_inputs=[agent_type, live_budget],
                    chatbot=gr.Chatbot(height=480, type="messages"),
                )
            else:
                gr.Markdown(
                    "### Live Agent Unavailable\n\n"
                    "No API key configured.\n\n"
                    "```bash\npip install tramontane\n"
                    "export MISTRAL_API_KEY=your_key\n"
                    "python examples/quickstart.py\n```\n\n"
                    "**Try Router Explorer and Fleet Intelligence tabs!**"
                )

    with gr.Accordion("Mistral Model Fleet (13 models)", open=False):
        fleet_table = gr.Dataframe(
            value=get_fleet_df(), interactive=False, wrap=True,
        )

    gr.Markdown(
        "---\n"
        "**TRAMONTANE v0.2.2** \u00b7 MIT License \u00b7 "
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
