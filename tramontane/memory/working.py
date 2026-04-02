"""Working memory — always-in-context blocks that the agent can edit.

Inspired by Letta's core memory: small, labeled blocks injected
into the system prompt. Agent edits them via set_working_block().
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass


@dataclass
class WorkingBlock:
    """A single working memory block."""

    id: str
    agent_id: str
    label: str
    content: str
    max_tokens: int = 500


class WorkingMemoryManager:
    """Manages working memory blocks in SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_blocks(self, agent_id: str) -> list[WorkingBlock]:
        """Get all working memory blocks for an agent."""
        rows = self._conn.execute(
            "SELECT id, agent_id, label, content, max_tokens "
            "FROM working_memory WHERE agent_id = ? ORDER BY label",
            (agent_id,),
        ).fetchall()
        return [
            WorkingBlock(
                id=row["id"],
                agent_id=row["agent_id"],
                label=row["label"],
                content=row["content"],
                max_tokens=row["max_tokens"],
            )
            for row in rows
        ]

    def set_block(
        self,
        agent_id: str,
        label: str,
        content: str,
        max_tokens: int = 500,
    ) -> None:
        """Set a working memory block (upsert by agent_id + label)."""
        existing = self._conn.execute(
            "SELECT id FROM working_memory WHERE agent_id = ? AND label = ?",
            (agent_id, label),
        ).fetchone()

        if existing:
            self._conn.execute(
                "UPDATE working_memory SET content = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (content, existing["id"]),
            )
        else:
            self._conn.execute(
                "INSERT INTO working_memory (id, agent_id, label, content, max_tokens) "
                "VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex[:12], agent_id, label, content, max_tokens),
            )
        self._conn.commit()

    def delete_block(self, agent_id: str, label: str) -> None:
        """Delete a working memory block."""
        self._conn.execute(
            "DELETE FROM working_memory WHERE agent_id = ? AND label = ?",
            (agent_id, label),
        )
        self._conn.commit()

    def format_for_prompt(self, agent_id: str) -> str:
        """Format all blocks as a string for system prompt injection."""
        blocks = self.get_blocks(agent_id)
        if not blocks:
            return ""
        parts = ["## Working Memory\n"]
        for block in blocks:
            # Truncate to max_tokens (~4 chars per token)
            max_chars = block.max_tokens * 4
            content = block.content[:max_chars]
            parts.append(f"### {block.label}\n{content}\n")
        return "\n".join(parts)
