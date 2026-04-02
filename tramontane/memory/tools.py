"""Memory tools — 5 agent-callable functions for memory management."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def create_memory_tools(memory: Any) -> list[Callable[..., Any]]:
    """Create 5 memory tools bound to a TramontaneMemory instance.

    These functions are designed for Tramontane's tool calling system.
    Add them to an Agent's tools list via memory_tools=True.
    """

    async def retain_memory(
        content: str, entity: str = "", category: str = "fact",
    ) -> str:
        """Store a new fact or insight in long-term memory.
        Call this when you learn something important that should be remembered."""
        memory_id = await memory.retain(
            content, entity=entity, category=category,
        )
        return f"Stored memory {memory_id}: {content[:100]}"

    async def recall_memory(query: str, top_k: int = 5) -> str:
        """Search long-term memory for relevant facts and experiences.
        Call this when you need context about a person, company, or past event."""
        results = await memory.recall(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        ctx: str = memory.format_context(results)
        return ctx

    async def reflect_on_memory(question: str) -> str:
        """Synthesize insights from multiple memories to answer a complex question.
        Call this for 'What patterns?' or 'What works best?' questions."""
        result: str = await memory.reflect(question)
        return result

    async def forget_memory(memory_id: str, reason: str = "") -> str:
        """Remove a memory. Use when information is outdated or deletion requested."""
        ok = await memory.forget(memory_id, reason=reason)
        return f"Memory {memory_id} {'forgotten' if ok else 'not found'}"

    async def update_memory(memory_id: str, new_content: str) -> str:
        """Correct or refresh an existing memory with new information."""
        ok = await memory.update(memory_id, new_content)
        return f"Memory {memory_id} {'updated' if ok else 'not found'}"

    return [
        retain_memory,
        recall_memory,
        reflect_on_memory,
        forget_memory,
        update_memory,
    ]
