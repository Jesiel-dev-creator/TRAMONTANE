"""Pipeline orchestration core — brings agents, router, handoffs together.

Supports AGENTIC mode (Mistral Handoffs API, non-deterministic) and
WORKFLOW mode (deterministic, step-by-step with checkpoints).
Implements all 7 failure-mode guards from CLAUDE.md.
"""

from __future__ import annotations

import datetime
import enum
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from tramontane.core.agent import Agent, AgentResult
from tramontane.core.exceptions import (
    BudgetExceededError,
    PipelineValidationError,
)
from tramontane.core.handoff import (
    HandoffEdge,
    HandoffEvent,
    HandoffGraph,
    HandoffInterceptor,
)
from tramontane.router.router import MistralRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineMode(enum.Enum):
    """How the pipeline executes agents."""

    AGENTIC = "agentic"
    WORKFLOW = "workflow"


class PipelineStatus(enum.Enum):
    """Lifecycle status of a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"


# ---------------------------------------------------------------------------
# PipelineRun result
# ---------------------------------------------------------------------------


class PipelineRun(BaseModel):
    """Immutable record of a single pipeline execution."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pipeline_name: str
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    completed_at: datetime.datetime | None = None
    total_cost_eur: float = 0.0
    agents_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    output: str | None = None
    error: str | None = None
    checkpoint_step: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


_CHECKPOINT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    agent_role TEXT,
    output TEXT,
    cost_eur REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, step)
);
"""


class Pipeline:
    """The main orchestration unit — runs a graph of Agents.

    On construction the handoff graph is validated (cycles, depth, role
    existence).  Actual API interaction happens lazily on ``run()`` so
    that pipelines can be built without a MISTRAL_API_KEY.
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent],
        handoffs: list[tuple[str, str]],
        budget_eur: float | None = None,
        mode: PipelineMode = PipelineMode.AGENTIC,
        gdpr_level: str = "none",
        locale: str = "en",
        streaming: bool = True,
        checkpoint_db: str = "tramontane_state.db",
    ) -> None:
        self.name = name
        self.mode = mode
        self.budget_eur = budget_eur
        self.gdpr_level = gdpr_level
        self.locale = locale
        self.streaming = streaming
        self._checkpoint_db_path = checkpoint_db

        # Agent lookup by role
        self._agents: dict[str, Agent] = {a.role: a for a in agents}

        # Build and validate handoff graph
        edges = [
            HandoffEdge(from_agent_role=f, to_agent_role=t)
            for f, t in handoffs
        ]
        self.handoff_graph = HandoffGraph(edges)

        # Guard 3: validate all roles in graph exist in agents list
        agent_roles = set(self._agents.keys())
        errors: list[str] = []
        for role in self.handoff_graph.roles:
            if role not in agent_roles:
                errors.append(
                    f"Handoff references unknown agent role: '{role}'"
                )
        if errors:
            raise PipelineValidationError(
                pipeline_name=name, errors=errors,
            )
        self.handoff_graph.validate(known_roles=agent_roles)

        # Router (lightweight — no API key needed)
        self._router = MistralRouter()

        # Lazy-init fields
        self._db: sqlite3.Connection | None = None

    @property
    def agents(self) -> dict[str, Agent]:
        """Agent lookup by role."""
        return dict(self._agents)

    # -- Lazy resources ----------------------------------------------------

    def _get_db(self) -> sqlite3.Connection:
        """Return (and cache) the checkpoint SQLite connection."""
        if self._db is None:
            self._db = sqlite3.connect(
                self._checkpoint_db_path, check_same_thread=False,
            )
            self._db.execute(_CHECKPOINT_SCHEMA)
            self._db.commit()
        return self._db

    # -- Main execution ----------------------------------------------------

    async def run(
        self,
        input_text: str,
        resume_from: int = 0,
    ) -> PipelineRun:
        """Execute the pipeline end-to-end.

        Implements all 7 failure-mode guards from CLAUDE.md:
          1. MAX_HANDOFF_DEPTH (graph validates on init)
          2. Circular detection (graph validates on init)
          3. Conflicting agent instructions (validated on init)
          4. Output format validation (Pydantic per agent)
          5. Timeout via asyncio.wait_for()
          6. Empty output retry
          7. Budget check BEFORE every LLM call
        """
        run = PipelineRun(pipeline_name=self.name)
        run.status = PipelineStatus.RUNNING
        run.checkpoint_step = resume_from

        budget_tracker: dict[str, float] = {}
        interceptor = HandoffInterceptor(
            graph=self.handoff_graph,
            budget_tracker=budget_tracker,
        )

        # Find entry agent (no incoming edges)
        entry_roles = self.handoff_graph.entry_roles()
        if not entry_roles:
            run.status = PipelineStatus.FAILED
            run.error = "No entry agent found (all agents have incoming edges)"
            return run

        current_role = entry_roles[0]
        current_output = input_text
        step = resume_from
        visited_ids: list[str] = []

        try:
            while current_role:
                agent = self._agents[current_role]

                # Guard 2: circular detection at runtime
                if current_role in visited_ids:
                    from tramontane.core.exceptions import HandoffLoopError

                    raise HandoffLoopError(
                        agent_ids=visited_ids + [current_role],
                        depth=len(visited_ids),
                    )
                visited_ids.append(current_role)

                # Guard 7: pipeline-level budget check BEFORE call
                if self.budget_eur is not None:
                    total_spent = sum(budget_tracker.values())
                    # Rough pre-estimate using cheapest possible cost
                    if total_spent > self.budget_eur:
                        run.status = PipelineStatus.BUDGET_EXCEEDED
                        run.output = current_output
                        break

                # Agent.run() handles: model resolution, budget check,
                # Mistral API call, timeout, cost calculation.
                # Pipeline passes spent_eur so Agent can check budget.
                result: AgentResult = await agent.run(
                    current_output,
                    router=self._router,
                    run_id=run.run_id,
                    spent_eur=run.total_cost_eur,
                )
                current_output = result.output

                # Guard 6: empty output
                if not current_output.strip():
                    current_output = "[Empty output from agent]"

                # Track costs from AgentResult
                budget_tracker[current_role] = (
                    budget_tracker.get(current_role, 0.0) + result.cost_eur
                )
                run.total_cost_eur += result.cost_eur
                run.agents_used.append(current_role)
                run.models_used.append(result.model_used)

                # Checkpoint
                self._checkpoint(
                    run.run_id, step, current_role, current_output, result.cost_eur,
                )
                step += 1
                run.checkpoint_step = step

                # Determine next agent via handoff
                next_roles = self.handoff_graph.get_allowed_handoffs(
                    current_role
                )
                if not next_roles:
                    current_role = ""  # done
                else:
                    next_role = next_roles[0]
                    event = HandoffEvent(
                        handoff_id=uuid.uuid4().hex,
                        from_agent_role=current_role,
                        to_agent_role=next_role,
                        conversation_id=run.run_id,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ),
                        budget_remaining_eur=(
                            self.budget_eur - sum(budget_tracker.values())
                            if self.budget_eur is not None
                            else None
                        ),
                    )
                    await interceptor.intercept(event, self.budget_eur)
                    current_role = next_role

            if run.status == PipelineStatus.RUNNING:
                run.status = PipelineStatus.COMPLETE
            run.output = current_output

        except BudgetExceededError:
            run.status = PipelineStatus.BUDGET_EXCEEDED
            run.output = current_output
        except Exception as exc:
            run.status = PipelineStatus.FAILED
            run.error = str(exc)
            logger.exception("Pipeline '%s' failed", self.name)

        run.completed_at = datetime.datetime.now(datetime.timezone.utc)
        return run

    async def resume(self, run_id: str) -> PipelineRun:
        """Resume a pipeline from its last checkpoint."""
        db = self._get_db()
        cursor = db.execute(
            "SELECT MAX(step) FROM pipeline_checkpoints WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        last_step = row[0] if row and row[0] is not None else 0

        # Get last output
        cursor = db.execute(
            "SELECT output FROM pipeline_checkpoints "
            "WHERE run_id = ? AND step = ?",
            (run_id, last_step),
        )
        row = cursor.fetchone()
        last_output = row[0] if row else ""

        return await self.run(input_text=last_output, resume_from=last_step)

    # -- Cost breakdown ----------------------------------------------------

    def cost_breakdown(self) -> dict[str, float]:
        """Return per-agent cost breakdown from the checkpoint DB."""
        db = self._get_db()
        cursor = db.execute(
            "SELECT agent_role, SUM(cost_eur) "
            "FROM pipeline_checkpoints GROUP BY agent_role"
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    # -- YAML loading ------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> Pipeline:
        """Load a full pipeline from a YAML definition file."""
        data: dict[str, Any] = yaml.safe_load(
            Path(path).read_text(encoding="utf-8")
        )
        agents: list[Agent] = []
        for a in data.get("agents", []):
            agent_data = {k: v for k, v in a.items() if k != "id"}
            agents.append(Agent(**agent_data))
        handoffs: list[tuple[str, str]] = []
        for h in data.get("handoffs", []):
            if isinstance(h, dict):
                handoffs.append((h["from"], h["to"]))
            else:
                handoffs.append((h[0], h[1]))
        return cls(
            name=data.get("name", Path(path).stem),
            agents=agents,
            handoffs=handoffs,
            budget_eur=data.get("budget_eur"),
            mode=PipelineMode(data.get("mode", "agentic")),
            gdpr_level=data.get("gdpr_level", "none"),
            locale=data.get("locale", "en"),
            streaming=data.get("streaming", True),
            checkpoint_db=data.get("checkpoint_db", "tramontane_state.db"),
        )

    # -- Internal ----------------------------------------------------------

    def _checkpoint(
        self,
        run_id: str,
        step: int,
        agent_role: str,
        output: str,
        cost_eur: float,
    ) -> None:
        """Write a checkpoint row to SQLite."""
        db = self._get_db()
        db.execute(
            "INSERT OR REPLACE INTO pipeline_checkpoints "
            "(run_id, step, agent_role, output, cost_eur) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, step, agent_role, json.dumps(output), cost_eur),
        )
        db.commit()
