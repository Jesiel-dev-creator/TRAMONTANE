"""Deterministic step-by-step workflow execution (Agno pattern).

Unlike AGENTIC mode, workflows execute a fixed sequence of steps with
checkpointing after each step.  On failure, resume from step N — never
restart from zero.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import sqlite3
import uuid
from collections import defaultdict
from typing import Any, Callable, TypeVar

from tramontane.core.agent import Agent

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Enums and data
# ---------------------------------------------------------------------------


class StepStatus(enum.Enum):
    """Lifecycle status of a single workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclasses.dataclass
class WorkflowStep:
    """A single step in a deterministic workflow."""

    step_id: str
    name: str
    fn: Callable[..., Any]
    agent: Agent
    depends_on: list[str] = dataclasses.field(default_factory=list)
    model_override: str | None = None
    budget_eur: float | None = None
    timeout_seconds: int | None = None
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: str | None = None
    cost_eur: float = 0.0


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def step(
    model: str = "auto",
    budget_eur: float | None = None,
    timeout_seconds: int | None = None,
    depends_on: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a method as a workflow step.

    Stores metadata as function attributes for inspection by
    the ``@workflow`` class decorator.
    """

    def decorator(fn: F) -> F:
        fn._step_model = model  # type: ignore[attr-defined]
        fn._step_budget_eur = budget_eur  # type: ignore[attr-defined]
        fn._step_timeout = timeout_seconds  # type: ignore[attr-defined]
        fn._step_depends_on = depends_on or []  # type: ignore[attr-defined]
        fn._is_step = True  # type: ignore[attr-defined]
        return fn

    return decorator


C = TypeVar("C")


def workflow(
    name: str,
    deterministic: bool = True,
) -> Callable[[C], C]:
    """Class decorator to register a class as a Tramontane workflow.

    Marks the class with metadata that Workflow can read to auto-discover
    steps decorated with ``@step``.
    """

    def decorator(cls: C) -> C:
        cls._workflow_name = name  # type: ignore[attr-defined]
        cls._workflow_deterministic = deterministic  # type: ignore[attr-defined]
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Checkpoint SQL
# ---------------------------------------------------------------------------

_WORKFLOW_CHECKPOINT_SCHEMA = """\
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_name TEXT,
    status TEXT,
    output TEXT,
    cost_eur REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, step_id)
);
"""


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow:
    """Deterministic, step-by-step pipeline with checkpoint/resume.

    Steps are executed in topological order.  After each step the result
    is persisted to SQLite so that a failed run can resume from the
    first non-COMPLETE step.
    """

    def __init__(
        self,
        name: str,
        steps: list[WorkflowStep],
        budget_eur: float | None = None,
        checkpoint_db: str = "tramontane_state.db",
    ) -> None:
        self.name = name
        self._steps: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        self._step_order = self._topological_sort(steps)
        self.budget_eur = budget_eur
        self._db_path = checkpoint_db
        self._db: sqlite3.Connection | None = None

    # -- Lazy DB -----------------------------------------------------------

    def _get_db(self) -> sqlite3.Connection:
        """Return (and cache) the checkpoint SQLite connection."""
        if self._db is None:
            self._db = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db.execute(_WORKFLOW_CHECKPOINT_SCHEMA)
            self._db.commit()
        return self._db

    # -- Execution ---------------------------------------------------------

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Execute all steps in topological order.

        Passes outputs of dependency steps as inputs to dependent steps.
        Checkpoints after each completion.
        """
        run_id = uuid.uuid4().hex
        results: dict[str, Any] = {}
        total_cost = 0.0

        for step_obj in self._step_order:
            if step_obj.status == StepStatus.COMPLETE:
                results[step_obj.name] = step_obj.output
                continue

            # Gather inputs from dependencies
            step_inputs: dict[str, Any] = dict(inputs)
            for dep in step_obj.depends_on:
                if dep in results:
                    step_inputs[dep] = results[dep]

            step_obj.status = StepStatus.RUNNING

            # Budget guard
            if self.budget_eur is not None and total_cost >= self.budget_eur:
                step_obj.status = StepStatus.SKIPPED
                continue

            retry_count = 0
            max_retries = step_obj.agent.max_retry_limit

            while retry_count <= max_retries:
                try:
                    coro = step_obj.fn(**step_inputs)

                    if step_obj.timeout_seconds:
                        output = await asyncio.wait_for(
                            coro, timeout=float(step_obj.timeout_seconds)
                        )
                    else:
                        output = await coro

                    step_obj.output = output
                    step_obj.status = StepStatus.COMPLETE
                    results[step_obj.name] = output

                    self._checkpoint_step(run_id, step_obj)
                    total_cost += step_obj.cost_eur
                    break

                except Exception as exc:
                    retry_count += 1
                    if retry_count > max_retries:
                        step_obj.status = StepStatus.FAILED
                        step_obj.error = str(exc)
                        self._checkpoint_step(run_id, step_obj)
                        raise
                    logger.warning(
                        "Step '%s' failed (retry %d/%d): %s",
                        step_obj.name, retry_count, max_retries, exc,
                    )

        return results

    async def resume(self, run_id: str, **inputs: Any) -> dict[str, Any]:
        """Resume a workflow from its last checkpoint.

        Loads checkpoint data, marks COMPLETE steps as done,
        and re-runs from the first non-COMPLETE step.
        """
        db = self._get_db()
        cursor = db.execute(
            "SELECT step_id, status, output "
            "FROM workflow_checkpoints WHERE run_id = ?",
            (run_id,),
        )
        for row in cursor.fetchall():
            step_id, status, output_json = row
            if step_id in self._steps and status == "complete":
                self._steps[step_id].status = StepStatus.COMPLETE
                self._steps[step_id].output = json.loads(output_json)

        # Re-sort and run (completed steps are skipped)
        return await self.run(**inputs)

    # -- Visualization -----------------------------------------------------

    def visualize(self) -> str:
        """Return a Mermaid diagram of step dependencies."""
        lines = ["graph TD"]
        for s in self._step_order:
            lines.append(f"  {s.step_id}[{s.name}]")
            for dep in s.depends_on:
                lines.append(f"  {dep} --> {s.step_id}")
        return "\n".join(lines)

    # -- Class method: from decorated class --------------------------------

    @classmethod
    def from_decorated_class(
        cls,
        workflow_cls: type,
        agent: Agent,
        budget_eur: float | None = None,
        checkpoint_db: str = "tramontane_state.db",
    ) -> Workflow:
        """Build a Workflow from a ``@workflow``-decorated class.

        Inspects the class for ``@step``-decorated methods and creates
        WorkflowStep instances from them.
        """
        name: str = getattr(workflow_cls, "_workflow_name", workflow_cls.__name__)
        instance = workflow_cls()
        steps: list[WorkflowStep] = []

        for attr_name in dir(instance):
            method = getattr(instance, attr_name, None)
            if callable(method) and getattr(method, "_is_step", False):
                steps.append(
                    WorkflowStep(
                        step_id=attr_name,
                        name=attr_name,
                        fn=method,
                        agent=agent,
                        depends_on=getattr(method, "_step_depends_on", []),
                        model_override=getattr(method, "_step_model", None),
                        budget_eur=getattr(method, "_step_budget_eur", None),
                        timeout_seconds=getattr(method, "_step_timeout", None),
                    )
                )

        return cls(
            name=name,
            steps=steps,
            budget_eur=budget_eur,
            checkpoint_db=checkpoint_db,
        )

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _topological_sort(steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """Kahn's algorithm topological sort by depends_on."""
        by_id: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        in_degree: dict[str, int] = defaultdict(int)
        for s in steps:
            in_degree.setdefault(s.step_id, 0)
            for dep in s.depends_on:
                in_degree[s.step_id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result: list[WorkflowStep] = []

        while queue:
            sid = queue.pop(0)
            if sid in by_id:
                result.append(by_id[sid])
            # Decrease in-degree for dependents
            for s in steps:
                if sid in s.depends_on:
                    in_degree[s.step_id] -= 1
                    if in_degree[s.step_id] == 0:
                        queue.append(s.step_id)

        if len(result) != len(steps):
            remaining = [s.step_id for s in steps if s not in result]
            logger.warning("Cycle detected in workflow steps: %s", remaining)

        return result

    def _checkpoint_step(self, run_id: str, step_obj: WorkflowStep) -> None:
        """Write a step checkpoint to SQLite."""
        db = self._get_db()
        db.execute(
            "INSERT OR REPLACE INTO workflow_checkpoints "
            "(run_id, step_id, step_name, status, output, cost_eur) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                step_obj.step_id,
                step_obj.name,
                step_obj.status.value,
                json.dumps(step_obj.output),
                step_obj.cost_eur,
            ),
        )
        db.commit()
