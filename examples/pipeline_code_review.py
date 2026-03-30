"""3-agent code review pipeline.

Loads the built-in code_review.yaml pipeline and runs it on a function.
Demonstrates: multi-agent handoff, automatic model routing, cost tracking.

Expected output:
    Status: complete
    Cost: ~EUR 0.002
    Agents: Senior Code Reviewer -> Security Auditor -> Technical Writer
"""

from __future__ import annotations

import anyio

from tramontane.core.pipeline import Pipeline


async def main() -> None:
    """Run the code review pipeline on a sample function."""
    pipeline = Pipeline.from_yaml("pipelines/code_review.yaml")
    pipeline.budget_eur = 0.10  # EUR ceiling

    result = await pipeline.run(
        input_text="""\
def calculate_discount(price, pct):
    discount = price * pct / 100
    return price - discount
"""
    )

    print(f"Status:  {result.status.value}")
    print(f"Cost:    EUR {result.total_cost_eur:.4f}")
    print(f"Agents:  {' -> '.join(result.agents_used)}")
    print(f"Models:  {result.models_used}")
    print(f"\n{(result.output or '')[:500]}")


if __name__ == "__main__":
    anyio.run(main)
