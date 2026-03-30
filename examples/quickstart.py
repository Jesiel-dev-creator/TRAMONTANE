"""Tramontane Quickstart — simplest possible agent call.

Runs a single agent with model="auto" (router picks the best model).
Requires MISTRAL_API_KEY in environment.

Expected output:
    Model: mistral-small (or similar, router decides)
    Cost: EUR 0.0001
    Output: Paris
"""

from __future__ import annotations

import anyio

from tramontane.core.agent import Agent
from tramontane.router.router import MistralRouter


async def main() -> None:
    """Run a single agent with automatic model routing."""
    agent = Agent(
        role="Geography Expert",
        goal="Answer geography questions accurately",
        backstory="You are a concise geography expert. One-word answers.",
        model="auto",
        budget_eur=0.01,
    )

    router = MistralRouter()
    result = await agent.run(
        "What is the capital of France? Answer in one word.",
        router=router,
    )

    print(f"Model: {result.model_used}")
    print(f"Cost:  EUR {result.cost_eur:.4f}")
    print(f"Output: {result.output.strip()}")


if __name__ == "__main__":
    anyio.run(main)
