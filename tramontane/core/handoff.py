"""Handoff graph and interceptor for Tramontane pipelines.

Wraps Mistral's Handoffs API with budget control, cycle detection,
and audit logging via handoff_execution="client" interception.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from typing import Callable

from pydantic import BaseModel, Field

from tramontane.core.exceptions import (
    BudgetExceededError,
    HandoffError,
    HandoffLoopError,
    PipelineValidationError,
)

logger = logging.getLogger(__name__)

MAX_HANDOFF_DEPTH = 10


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class HandoffEdge(BaseModel):
    """A directed edge in the handoff graph."""

    from_agent_role: str
    to_agent_role: str
    condition: str | None = None


class HandoffEvent(BaseModel):
    """An event emitted when a handoff occurs at runtime."""

    handoff_id: str
    from_agent_role: str
    to_agent_role: str
    conversation_id: str
    timestamp: datetime.datetime
    budget_remaining_eur: float | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HandoffGraph
# ---------------------------------------------------------------------------


class HandoffGraph:
    """Directed graph of agent-to-agent handoffs with validation.

    Validates no cycles, max depth <= MAX_HANDOFF_DEPTH, and exposes
    helpers for allowed-handoff lookup and Mermaid diagram generation.
    """

    def __init__(self, edges: list[HandoffEdge]) -> None:
        self._edges = edges
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._all_roles: set[str] = set()

        for edge in edges:
            self._adjacency[edge.from_agent_role].append(edge.to_agent_role)
            self._all_roles.add(edge.from_agent_role)
            self._all_roles.add(edge.to_agent_role)

        # Validate structural integrity on construction
        self._check_cycles()
        self._check_max_depth()

    # -- Public API --------------------------------------------------------

    def validate(self, known_roles: set[str] | None = None) -> None:
        """Full validation including agent existence checks.

        Raises PipelineValidationError with all issues found.
        """
        errors: list[str] = []

        # Re-run structural checks (cycle + depth)
        try:
            self._check_cycles()
        except HandoffLoopError as exc:
            errors.append(str(exc))

        try:
            self._check_max_depth()
        except HandoffLoopError as exc:
            errors.append(str(exc))

        # Check that every role referenced in edges is a known agent
        if known_roles is not None:
            for role in self._all_roles:
                if role not in known_roles:
                    errors.append(f"Unknown agent role in handoff graph: '{role}'")

        if errors:
            raise PipelineValidationError(
                pipeline_name="handoff_graph",
                errors=errors,
            )

    def get_allowed_handoffs(self, from_role: str) -> list[str]:
        """Return list of valid next agents for a given role."""
        return list(self._adjacency.get(from_role, []))

    def depth_from(self, start_role: str) -> int:
        """Return the max depth (longest path) reachable from start_role."""
        visited: set[str] = set()

        def _dfs(role: str) -> int:
            if role in visited:
                return 0
            visited.add(role)
            children = self._adjacency.get(role, [])
            if not children:
                return 0
            max_child = max(_dfs(c) for c in children)
            visited.discard(role)
            return 1 + max_child

        return _dfs(start_role)

    def to_mermaid(self) -> str:
        """Return a Mermaid diagram string for docs/dashboard."""
        lines = ["graph LR"]
        for edge in self._edges:
            f = edge.from_agent_role
            t = edge.to_agent_role
            label = edge.condition or "handoff"
            lines.append(f"  {f}[{f}] -->|{label}| {t}[{t}]")
        return "\n".join(lines)

    @property
    def roles(self) -> set[str]:
        """All roles referenced in the graph."""
        return set(self._all_roles)

    @property
    def edges(self) -> list[HandoffEdge]:
        """All edges in the graph."""
        return list(self._edges)

    def entry_roles(self) -> list[str]:
        """Roles with no incoming edges (pipeline entry points)."""
        targets = {e.to_agent_role for e in self._edges}
        return [r for r in self._all_roles if r not in targets]

    # -- Internal checks ---------------------------------------------------

    def _check_cycles(self) -> None:
        """Detect cycles using DFS with coloring."""
        white, gray, black = 0, 1, 2
        color: dict[str, int] = {r: white for r in self._all_roles}
        path: list[str] = []

        def _visit(role: str) -> None:
            color[role] = gray
            path.append(role)
            for neighbor in self._adjacency.get(role, []):
                if color[neighbor] == gray:
                    cycle_start = path.index(neighbor)
                    raise HandoffLoopError(
                        agent_ids=path[cycle_start:],
                        depth=len(path),
                    )
                if color[neighbor] == white:
                    _visit(neighbor)
            path.pop()
            color[role] = black

        for role in self._all_roles:
            if color[role] == white:
                _visit(role)

    def _check_max_depth(self) -> None:
        """Ensure no path exceeds MAX_HANDOFF_DEPTH."""
        for role in self.entry_roles() or list(self._all_roles):
            depth = self.depth_from(role)
            if depth > MAX_HANDOFF_DEPTH:
                raise HandoffLoopError(
                    agent_ids=[role],
                    depth=depth,
                )


# ---------------------------------------------------------------------------
# HandoffInterceptor
# ---------------------------------------------------------------------------


class HandoffInterceptor:
    """Intercepts handoffs in client-mode execution.

    Called with handoff_execution="client" — checks budget, graph validity,
    and fires audit callbacks before allowing or blocking a handoff.
    """

    def __init__(
        self,
        graph: HandoffGraph,
        budget_tracker: dict[str, float],
        audit_fn: Callable[[HandoffEvent], None] | None = None,
    ) -> None:
        self._graph = graph
        self._budget_tracker = budget_tracker
        self._audit_fn = audit_fn

    async def intercept(
        self,
        event: HandoffEvent,
        pipeline_budget_eur: float | None = None,
    ) -> bool:
        """Intercept a handoff event. Returns True to allow, False to block.

        Raises BudgetExceededError or HandoffError when hard constraints
        are violated.
        """
        # Check: is handoff allowed by graph?
        allowed = self._graph.get_allowed_handoffs(event.from_agent_role)
        if event.to_agent_role not in allowed:
            raise HandoffError(
                from_agent=event.from_agent_role,
                to_agent=event.to_agent_role,
                reason=(
                    f"not in allowed handoffs: {allowed}"
                ),
            )

        # Check: budget
        if pipeline_budget_eur is not None:
            total_spent = sum(self._budget_tracker.values())
            if total_spent > pipeline_budget_eur:
                raise BudgetExceededError(
                    budget_eur=pipeline_budget_eur,
                    spent_eur=total_spent,
                    pipeline_name=event.conversation_id,
                )

        # Audit
        if self._audit_fn is not None:
            self._audit_fn(event)

        logger.debug(
            "Handoff allowed: %s -> %s (conversation %s)",
            event.from_agent_role,
            event.to_agent_role,
            event.conversation_id,
        )
        return True
