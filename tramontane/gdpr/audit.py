"""Append-only audit vault for Tramontane.

Writes to tramontane_audit table. NEVER deletes rows.
Rich CLI output follows the EU Premium / Night Authority design system.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parents[1] / "memory" / "schema.sql"

# -- EU Premium Design System (Rich styles) --------------------------------
_CYAN = "#00D4EE"
_EMBER = "#FF6B35"
_FROST = "#DCE9F5"
_STORM = "#4A6480"
_OK = "#22D68A"
_WARN = "#FFB020"
_ERR = "#FF4560"
_RIM = "#1C2E42"

_console = Console()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single audit log entry — matches tramontane_audit schema."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    pipeline_name: str | None = None
    agent_role: str | None = None
    action_type: str = "llm_call"
    model_used: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    gdpr_sensitivity: str = "none"
    pii_detected: bool = False
    pii_redacted: bool = False
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# AuditVault
# ---------------------------------------------------------------------------


class AuditVault:
    """Append-only audit log backed by SQLite.

    Auto-creates DB/tables on first write. Uses WAL mode.
    NEVER updates or deletes rows.
    """

    def __init__(self, db_path: str = "tramontane.db") -> None:
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None

    def _get_db(self) -> sqlite3.Connection:
        """Return (and cache) the SQLite connection, creating schema if needed."""
        if self._db is not None:
            return self._db

        self._db = sqlite3.connect(self._db_path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")

        cursor = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='tramontane_audit'"
        )
        if cursor.fetchone() is None:
            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            self._db.executescript(schema_sql)
            logger.info("Tramontane audit schema created in %s", self._db_path)

        return self._db

    # -- Logging (append-only) ---------------------------------------------

    async def log(
        self,
        run_id: str,
        pipeline_name: str,
        agent_role: str,
        action_type: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        cost_eur: float,
        gdpr_sensitivity: str = "none",
        pii_detected: bool = False,
        pii_redacted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Insert an audit entry. NEVER updates or deletes."""
        db = self._get_db()
        entry = AuditEntry(
            run_id=run_id,
            pipeline_name=pipeline_name,
            agent_role=agent_role,
            action_type=action_type,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_eur=cost_eur,
            gdpr_sensitivity=gdpr_sensitivity,
            pii_detected=pii_detected,
            pii_redacted=pii_redacted,
            metadata=metadata,
        )

        db.execute(
            "INSERT INTO tramontane_audit "
            "(id, run_id, pipeline_name, agent_role, action_type, "
            "model_used, input_tokens, output_tokens, cost_eur, "
            "gdpr_sensitivity, pii_detected, pii_redacted, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.run_id,
                entry.pipeline_name,
                entry.agent_role,
                entry.action_type,
                entry.model_used,
                entry.input_tokens,
                entry.output_tokens,
                entry.cost_eur,
                entry.gdpr_sensitivity,
                entry.pii_detected,
                entry.pii_redacted,
                entry.timestamp.isoformat(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        db.commit()
        return entry

    def log_sync(
        self,
        run_id: str,
        pipeline_name: str,
        agent_role: str,
        action_type: str,
        model_used: str,
        input_tokens: int,
        output_tokens: int,
        cost_eur: float,
        gdpr_sensitivity: str = "none",
        pii_detected: bool = False,
        pii_redacted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Synchronous wrapper for log(). Do not call from async context."""
        from tramontane.core._sync import run_sync

        return run_sync(
            self.log(
                run_id=run_id,
                pipeline_name=pipeline_name,
                agent_role=agent_role,
                action_type=action_type,
                model_used=model_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_eur=cost_eur,
                gdpr_sensitivity=gdpr_sensitivity,
                pii_detected=pii_detected,
                pii_redacted=pii_redacted,
                metadata=metadata,
            )
        )

    # -- Queries -----------------------------------------------------------

    async def get_run(self, run_id: str) -> list[AuditEntry]:
        """Get all audit entries for a run."""
        db = self._get_db()
        cursor = db.execute(
            "SELECT * FROM tramontane_audit WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def get_pipeline(
        self,
        pipeline_name: str,
        since: datetime.datetime | None = None,
    ) -> list[AuditEntry]:
        """Get audit entries for a pipeline, optionally filtered by time."""
        db = self._get_db()
        if since is not None:
            cursor = db.execute(
                "SELECT * FROM tramontane_audit "
                "WHERE pipeline_name = ? AND timestamp >= ? ORDER BY timestamp",
                (pipeline_name, since.isoformat()),
            )
        else:
            cursor = db.execute(
                "SELECT * FROM tramontane_audit "
                "WHERE pipeline_name = ? ORDER BY timestamp",
                (pipeline_name,),
            )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def total_cost(self, run_id: str) -> float:
        """Total EUR cost for a run."""
        db = self._get_db()
        cursor = db.execute(
            "SELECT COALESCE(SUM(cost_eur), 0.0) FROM tramontane_audit "
            "WHERE run_id = ?",
            (run_id,),
        )
        result: float = cursor.fetchone()[0]
        return result

    async def cost_by_model(self, run_id: str) -> dict[str, float]:
        """Per-model cost breakdown for a run."""
        db = self._get_db()
        cursor = db.execute(
            "SELECT model_used, SUM(cost_eur) FROM tramontane_audit "
            "WHERE run_id = ? GROUP BY model_used",
            (run_id,),
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    # -- Rich CLI display (EU Premium design system) -----------------------

    def display_run(self, run_id: str) -> None:
        """Display run audit as a Rich table — EU Premium style."""
        from tramontane.core._sync import run_sync

        entries = run_sync(self.get_run(run_id))
        if not entries:
            _console.print(f"[{_STORM}]No audit entries for run {run_id}[/]")
            return

        table = Table(
            title=f"Audit: {run_id[:12]}...",
            title_style=f"bold {_CYAN}",
            box=box.MINIMAL_HEAVY_HEAD,
            header_style=f"bold {_CYAN}",
            border_style=f"dim {_RIM}",
            show_footer=True,
        )
        table.add_column("Timestamp", style=_STORM)
        table.add_column("Agent", style=_FROST)
        table.add_column("Model", style=f"italic {_CYAN}")
        table.add_column("Tokens In", justify="right", style=_FROST)
        table.add_column("Tokens Out", justify="right", style=_FROST)
        table.add_column("Cost", justify="right", style=f"bold {_WARN}")
        table.add_column("Action", style=_FROST)

        total_cost = 0.0
        for e in entries:
            total_cost += e.cost_eur
            ts = e.timestamp.strftime("%H:%M:%S") if e.timestamp else ""
            table.add_row(
                ts,
                e.agent_role or "-",
                e.model_used or "-",
                str(e.input_tokens),
                str(e.output_tokens),
                f"€{e.cost_eur:.4f}",
                e.action_type,
            )

        # Footer with total
        table.columns[5].footer = f"€{total_cost:.4f}"
        table.columns[5].footer_style = f"bold {_WARN}"
        table.columns[0].footer = "TOTAL"
        table.columns[0].footer_style = f"bold {_FROST}"

        _console.print(table)

    def display_cost_breakdown(self, run_id: str) -> None:
        """Display per-model cost breakdown — EU Premium style."""
        from tramontane.core._sync import run_sync

        costs = run_sync(self.cost_by_model(run_id))
        if not costs:
            _console.print(f"[{_STORM}]No cost data for run {run_id}[/]")
            return

        max_cost = max(costs.values()) if costs else 1.0
        table = Table(
            title="Cost Breakdown by Model",
            title_style=f"bold {_CYAN}",
            box=box.MINIMAL_HEAVY_HEAD,
            header_style=f"bold {_CYAN}",
            border_style=f"dim {_RIM}",
        )
        table.add_column("Model", style=f"italic {_CYAN}")
        table.add_column("Cost", justify="right", style=f"bold {_WARN}")
        table.add_column("Bar", min_width=30)

        for model, cost in sorted(costs.items(), key=lambda x: -x[1]):
            bar_len = int((cost / max_cost) * 25) if max_cost > 0 else 0
            bar = f"[{_CYAN}]{'█' * bar_len}[/][{_STORM}]{'░' * (25 - bar_len)}[/]"
            table.add_row(model, f"€{cost:.4f}", bar)

        _console.print(table)

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        """Convert a sqlite3.Row to an AuditEntry."""
        meta_raw = row["metadata"]
        metadata: dict[str, Any] | None = None
        if meta_raw:
            metadata = json.loads(meta_raw)

        return AuditEntry(
            id=row["id"],
            run_id=row["run_id"],
            pipeline_name=row["pipeline_name"],
            agent_role=row["agent_role"],
            action_type=row["action_type"],
            model_used=row["model_used"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_eur=row["cost_eur"],
            gdpr_sensitivity=row["gdpr_sensitivity"],
            pii_detected=bool(row["pii_detected"]),
            pii_redacted=bool(row["pii_redacted"]),
            metadata=metadata,
        )
