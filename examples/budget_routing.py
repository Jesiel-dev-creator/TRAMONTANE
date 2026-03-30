"""Budget-aware routing with quality floors.

Shows how the router selects models based on budget constraints
while enforcing minimum quality per task type.

No API key needed — uses OFFLINE classifier mode.

Expected output:
    Task: Write a Python parser       -> devstral-small  EUR 0.0001
    Task: Explain quantum computing   -> magistral-small EUR 0.0004
    Task: List EU countries            -> ministral-7b   EUR 0.0000
"""

from __future__ import annotations

import anyio

from tramontane.router.router import MistralRouter


async def main() -> None:
    """Demonstrate budget-aware routing with quality floors."""
    router = MistralRouter()

    tasks = [
        ("Write a Python function to parse JSON", 0.05, "en"),
        ("Explain step by step why quantum computing breaks RSA", 0.01, "en"),
        ("List all EU member states", 0.005, "en"),
        ("Redige un email de prospection en francais", 0.05, "fr"),
        ("Analyze this contract for legal risks and obligations", 0.10, "fr"),
    ]

    for prompt, budget, locale in tasks:
        decision = await router.route(
            prompt=prompt,
            agent_budget_eur=budget,
            locale=locale,
        )
        downgrade = " [downgraded]" if decision.downgrade_applied else ""
        print(
            f"  {prompt[:45]:<47} -> {decision.primary_model:<18} "
            f"EUR {decision.estimated_cost_eur:.4f}{downgrade}"
        )


if __name__ == "__main__":
    anyio.run(main)
