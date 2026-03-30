"""GDPR compliance reports — Article 30 processing records and more.

Generates structured reports from the audit vault for compliance
officers. Rich CLI output follows the EU Premium design system.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tramontane.gdpr.audit import AuditVault
from tramontane.memory.longterm import LongTermMemory

logger = logging.getLogger(__name__)

# EU Premium palette
_CYAN = "#00D4EE"
_FROST = "#DCE9F5"
_STORM = "#4A6480"
_WARN = "#FFB020"
_OK = "#22D68A"
_ERR = "#FF4560"
_RIM = "#1C2E42"

_console = Console()


class GDPRReporter:
    """Generates GDPR compliance reports from the audit vault.

    Supports Article 30 processing records, data inventory,
    and erasure reports.
    """

    def __init__(
        self,
        audit_vault: AuditVault | None = None,
        memory: LongTermMemory | None = None,
    ) -> None:
        self._audit = audit_vault or AuditVault()
        self._memory = memory or LongTermMemory()

    # -- Article 30: Record of Processing Activities -----------------------

    async def article_30_report(
        self,
        pipeline_name: str | None = None,
        since: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Generate Article 30 processing records.

        Returns a structured dict suitable for JSON export or display.
        """
        if pipeline_name:
            entries = await self._audit.get_pipeline(pipeline_name, since)
        else:
            entries = await self._audit.get_pipeline("", since)

        total_cost = sum(e.cost_eur for e in entries)
        models_used = list({e.model_used for e in entries if e.model_used})
        pii_events = [e for e in entries if e.pii_detected]

        report: dict[str, Any] = {
            "report_type": "GDPR Article 30 — Record of Processing Activities",
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "controller": "Bleucommerce SAS, Orleans, France",
            "pipeline": pipeline_name or "all",
            "period": {
                "from": since.isoformat() if since else "inception",
                "to": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            },
            "processing_activities": {
                "total_operations": len(entries),
                "models_used": models_used,
                "total_cost_eur": round(total_cost, 4),
            },
            "data_protection": {
                "pii_events": len(pii_events),
                "pii_redacted": sum(1 for e in pii_events if e.pii_redacted),
                "gdpr_sensitivity_breakdown": self._sensitivity_breakdown(
                    entries
                ),
            },
            "data_residency": {
                "processing_region": "EU (fr-par)",
                "provider": "Mistral AI (La Plateforme)",
                "subprocessors": ["Scaleway EU-west-1 Paris"],
            },
        }
        return report

    def article_30_report_sync(
        self,
        pipeline_name: str | None = None,
        since: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Synchronous wrapper for article_30_report()."""
        from tramontane.core._sync import run_sync

        return run_sync(self.article_30_report(pipeline_name, since))

    # -- Erasure report ----------------------------------------------------

    async def erasure_report(self) -> dict[str, Any]:
        """Generate a report of all GDPR erasure events."""
        db = self._memory._get_db()
        cursor = db.execute(
            "SELECT * FROM tramontane_erasure_log ORDER BY erased_at DESC"
        )
        events = [dict(row) for row in cursor.fetchall()]

        return {
            "report_type": "GDPR Article 17 — Erasure Log",
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "total_erasure_requests": len(events),
            "total_records_erased": sum(
                e.get("erased_count", 0) for e in events
            ),
            "events": events,
        }

    # -- Data inventory ----------------------------------------------------

    async def data_inventory(self) -> dict[str, Any]:
        """Generate a data inventory for compliance review."""
        stats = self._memory.stats()
        return {
            "report_type": "Data Inventory",
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "memory_store": stats,
            "data_categories": [
                "conversation_history",
                "entity_facts",
                "user_preferences",
                "pipeline_outputs",
            ],
            "retention_policy": "configurable per-entry TTL, manual GDPR erasure",
            "storage_location": "local SQLite (EU-hosted infrastructure)",
        }

    # -- Export to JSON ----------------------------------------------------

    def export_json(self, report: dict[str, Any]) -> str:
        """Export a report as formatted JSON."""
        return json.dumps(report, indent=2, default=str, ensure_ascii=False)

    # -- Rich CLI display --------------------------------------------------

    def display_article_30(
        self,
        pipeline_name: str | None = None,
        since: datetime.datetime | None = None,
    ) -> None:
        """Display Article 30 report with EU Premium styling."""
        report = self.article_30_report_sync(pipeline_name, since)

        _console.print()
        _console.print(
            Panel(
                f"[bold {_CYAN}]GDPR Article 30 — Record of Processing Activities[/]",
                border_style=_RIM,
                padding=(0, 2),
            )
        )

        # Summary table
        table = Table(
            box=box.MINIMAL_HEAVY_HEAD,
            header_style=f"bold {_CYAN}",
            border_style=f"dim {_RIM}",
        )
        table.add_column("Field", style=_STORM)
        table.add_column("Value", style=_FROST)

        proc = report["processing_activities"]
        data = report["data_protection"]
        table.add_row("Controller", report["controller"])
        table.add_row("Pipeline", report["pipeline"])
        table.add_row(
            "Period",
            f"{report['period']['from']} → {report['period']['to']}",
        )
        table.add_row("Total Operations", str(proc["total_operations"]))
        table.add_row("Models Used", ", ".join(proc["models_used"]))
        table.add_row(
            "Total Cost",
            f"[bold {_WARN}]€{proc['total_cost_eur']:.4f}[/]",
        )
        table.add_row("PII Events", str(data["pii_events"]))
        table.add_row("PII Redacted", str(data["pii_redacted"]))

        sens = data["gdpr_sensitivity_breakdown"]
        if sens.get("high", 0) > 0:
            pii_str = f"[bold {_ERR}]high: {sens['high']}[/]"
        elif sens.get("low", 0) > 0:
            pii_str = f"[{_WARN}]low: {sens['low']}[/]"
        else:
            pii_str = f"[{_OK}]none detected[/]"
        table.add_row("Sensitivity", pii_str)

        table.add_row(
            "Data Residency",
            f"{report['data_residency']['processing_region']} "
            f"({report['data_residency']['provider']})",
        )

        _console.print(table)
        _console.print()

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _sensitivity_breakdown(
        entries: list[Any],
    ) -> dict[str, int]:
        """Count entries by GDPR sensitivity level."""
        counts: dict[str, int] = {"none": 0, "low": 0, "high": 0}
        for e in entries:
            level = getattr(e, "gdpr_sensitivity", "none") or "none"
            if level in counts:
                counts[level] += 1
        return counts
