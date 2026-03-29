"""Tramontane CLI — EU Premium design throughout.

The primary interface for running pipelines, inspecting models,
managing audit trails, and initialising the local environment.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import tramontane

# ---------------------------------------------------------------------------
# EU Premium palette
# ---------------------------------------------------------------------------
_CYAN = "#00D4EE"
_EMBER = "#FF6B35"
_FROST = "#DCE9F5"
_STORM = "#4A6480"
_OK = "#22D68A"
_WARN = "#FFB020"
_ERR = "#FF4560"
_RIM = "#1C2E42"

console = Console()

app = typer.Typer(
    name="tramontane",
    help="Mistral-native agent orchestration framework",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _show_banner() -> None:
    """Display the EU Premium startup banner."""
    console.print()
    console.print(Rule(title="TRAMONTANE", style=f"bold {_CYAN}"))
    console.print(
        Panel(
            f"[bold {_FROST}]Mistral-native agent orchestration[/]\n"
            f"[{_STORM}]v{tramontane.__version__}"
            f" \u00b7 Bleucommerce SAS \u00b7 Orl\u00e9ans, France[/]\n"
            f"[{_STORM}]MIT License \u00b7 EU Sovereign \u00b7 Scaleway-ready[/]",
            border_style=f"dim {_RIM}",
            padding=(0, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Callback (--version, --verbose, no-subcommand)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging"),
) -> None:
    """Tramontane — Mistral-native agent orchestration framework."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if version or ctx.invoked_subcommand is None:
        _show_banner()
        if version:
            raise typer.Exit()


# ---------------------------------------------------------------------------
# tramontane run
# ---------------------------------------------------------------------------


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Path to pipeline YAML"),
    input_text: Optional[str] = typer.Option(None, "--input", "-i", help="Input text"),
    file_path: Optional[Path] = typer.Option(None, "--file", "-f", help="Input file"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget in EUR"),
    local: bool = typer.Option(False, "--local", help="Use local Ollama models"),
    voice: bool = typer.Option(False, "--voice", help="Voice input mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    gdpr: str = typer.Option("none", "--gdpr", help="GDPR level: none|standard|strict"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run a pipeline from a YAML definition."""
    from tramontane.core.pipeline import Pipeline

    # Resolve input
    text = input_text or ""
    if file_path and file_path.exists():
        text = file_path.read_text(encoding="utf-8")

    if voice and input_text:
        from tramontane.voice.gateway import VoiceGateway

        gw = VoiceGateway()
        if gw.is_available():
            result = gw.transcribe_file_sync(input_text)
            text = result.transcript
            console.print(f"[{_CYAN}]Transcribed:[/] {text[:80]}...")
        else:
            console.print(f"[{_ERR}]Voice unavailable — MISTRAL_API_KEY not set[/]")
            raise typer.Exit(1)

    if not text:
        console.print(f"[{_ERR}]No input provided. Use --input, --file, or --voice[/]")
        raise typer.Exit(1)

    # Load pipeline
    try:
        pipe = Pipeline.from_yaml(pipeline)
    except Exception as exc:
        console.print(f"[{_ERR}]Failed to load pipeline: {exc}[/]")
        raise typer.Exit(1) from exc

    if budget is not None:
        pipe.budget_eur = budget

    # Pre-run info
    agents = ", ".join(a.role for a in pipe._agents.values())
    console.print(
        Panel(
            f"[bold {_FROST}]{pipe.name}[/]\n"
            f"[{_STORM}]Agents:[/] [{_CYAN}]{agents}[/]\n"
            f"[{_STORM}]Budget:[/] [{_WARN}]"
            f"{'EUR ' + f'{pipe.budget_eur:.4f}' if pipe.budget_eur else 'unlimited'}[/]\n"
            f"[{_STORM}]GDPR:[/] {gdpr}",
            title="Pipeline",
            title_align="left",
            border_style=f"dim {_RIM}",
        )
    )

    # Run
    console.print(f"\n[{_CYAN}]Running pipeline...[/]")
    pipe_run = asyncio.run(pipe.run(input_text=text))

    # Summary
    status_color = _OK if pipe_run.status.value == "complete" else _ERR
    console.print(
        Panel(
            f"[bold {_FROST}]Pipeline {pipe_run.status.value.upper()}[/]\n"
            f"[{_STORM}]Pipeline:[/]  {pipe.name}\n"
            f"[{_STORM}]Total cost:[/] [bold {_WARN}]EUR {pipe_run.total_cost_eur:.4f}[/]\n"
            f"[{_STORM}]Models:[/]    [{_CYAN}]{', '.join(pipe_run.models_used)}[/]",
            title="Complete",
            title_align="left",
            border_style=f"dim {status_color}",
        )
    )

    if output and pipe_run.output:
        output.write_text(pipe_run.output, encoding="utf-8")
        console.print(f"[{_OK}]Output written to {output}[/]")


# ---------------------------------------------------------------------------
# tramontane models
# ---------------------------------------------------------------------------


@app.command()
def models() -> None:
    """Display the Mistral model fleet."""
    from tramontane.router.models import MISTRAL_MODELS

    console.print()
    console.print(Rule(title="MISTRAL MODEL FLEET", style=f"bold {_CYAN}"))
    console.print()

    table = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        header_style=f"bold {_CYAN}",
        border_style=f"dim {_RIM}",
    )
    table.add_column("Model", style=_FROST)
    table.add_column("Tier", justify="center")
    table.add_column("Strengths", style=_STORM, max_width=30)
    table.add_column("EUR/1M in", justify="right", style=f"bold {_WARN}")
    table.add_column("EUR/1M out", justify="right", style=f"bold {_WARN}")
    table.add_column("Local")
    table.add_column("Modality", style=_STORM)

    for alias, m in MISTRAL_MODELS.items():
        # Tier styling: 0-1 dim, 2 normal, 3-4 bold
        if m.tier <= 1:
            tier_str = f"[dim]{m.tier}[/]"
            name_str = f"[dim {_FROST}]{alias}[/]"
        elif m.tier >= 3:
            tier_str = f"[bold {_EMBER}]{m.tier}[/]"
            name_str = f"[bold {_CYAN}]{alias}[/]"
        else:
            tier_str = f"[{_FROST}]{m.tier}[/]"
            name_str = f"[{_CYAN}]{alias}[/]"

        local_str = f"[{_OK}]ollama[/]" if m.local_ollama else f"[dim {_STORM}]\u2014[/]"

        table.add_row(
            name_str,
            tier_str,
            ", ".join(m.strengths[:3]),
            f"\u20ac{m.cost_per_1m_input_eur:.2f}",
            f"\u20ac{m.cost_per_1m_output_eur:.2f}",
            local_str,
            m.modality,
        )

    console.print(table)
    console.print(
        f"\n  [{_STORM}]Prices in EUR \u00b7 Source: Mistral AI \u00b7 Updated March 2026[/]\n"
    )


# ---------------------------------------------------------------------------
# tramontane watch
# ---------------------------------------------------------------------------


@app.command()
def watch() -> None:
    """Watch active pipeline runs in real time."""
    console.print()
    console.print(Rule(title="PIPELINE WATCH", style=f"bold {_CYAN}"))
    console.print(
        f"\n  [{_STORM}]No active runs. Start a pipeline with:[/] "
        f"[bold {_CYAN}]tramontane run <pipeline.yaml> --input '...'[/]\n"
    )


# ---------------------------------------------------------------------------
# tramontane audit
# ---------------------------------------------------------------------------


@app.command()
def audit(
    pipeline: Optional[str] = typer.Option(None, help="Filter by pipeline name"),
    run_id: Optional[str] = typer.Option(None, "--run", help="Filter by run ID"),
    export: Optional[str] = typer.Option(
        None, help="Export format: article30 | json | markdown",
    ),
) -> None:
    """View or export audit trails."""
    from tramontane.gdpr.audit import AuditVault
    from tramontane.gdpr.reports import GDPRReporter

    vault = AuditVault()

    if export == "article30":
        reporter = GDPRReporter(audit_vault=vault)
        report = reporter.article_30_report_sync(pipeline_name=pipeline)
        console.print_json(data=report)
    elif export == "json":
        if run_id:
            entries = asyncio.run(vault.get_run(run_id))
            console.print_json(
                data=[e.model_dump(mode="json") for e in entries]
            )
        else:
            console.print(f"[{_ERR}]--run required for JSON export[/]")
    elif run_id:
        vault.display_run(run_id)
    else:
        console.print(
            f"\n  [{_STORM}]Usage:[/] tramontane audit --run <run_id>\n"
            f"  [{_STORM}]Export:[/] tramontane audit --export article30\n"
        )


# ---------------------------------------------------------------------------
# tramontane init
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_cmd(
    db_path: str = typer.Option("tramontane.db", help="Database path"),
) -> None:
    """Initialise the Tramontane database and verify health."""
    from tramontane.gdpr.audit import AuditVault
    from tramontane.memory.longterm import LongTermMemory

    console.print()
    console.print(Rule(title="TRAMONTANE INIT", style=f"bold {_CYAN}"))
    console.print()

    checks: list[tuple[str, bool, str]] = []

    # Database
    try:
        mem = LongTermMemory(db_path=db_path)
        mem._get_db()
        checks.append(("Database created", True, db_path))
    except Exception as exc:
        checks.append(("Database created", False, str(exc)))

    # Schema
    try:
        mem = LongTermMemory(db_path=db_path)
        db = mem._get_db()
        cursor = db.execute(
            "SELECT version FROM tramontane_schema ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        version = row[0] if row else "unknown"
        checks.append(("Schema applied", True, f"version {version}"))
    except Exception as exc:
        checks.append(("Schema applied", False, str(exc)))

    # Audit vault
    try:
        vault = AuditVault(db_path=db_path)
        vault._get_db()
        checks.append(("Audit vault", True, "ready"))
    except Exception as exc:
        checks.append(("Audit vault", False, str(exc)))

    # Memory store
    try:
        stats = mem.stats()
        checks.append((
            "Memory store",
            True,
            f"{stats['total_entries']} entries, {stats['db_size_mb']}MB",
        ))
    except Exception as exc:
        checks.append(("Memory store", False, str(exc)))

    # Display results
    for label, ok, detail in checks:
        if ok:
            console.print(f"  [{_OK}]\u2713[/] [{_FROST}]{label}[/] [{_STORM}]{detail}[/]")
        else:
            console.print(f"  [{_ERR}]\u2717[/] [{_FROST}]{label}[/] [{_ERR}]{detail}[/]")

    # .env.example
    env_example = Path(".env.example")
    if not env_example.exists():
        src = Path(__file__).parents[1] / ".." / ".env.example"
        if src.exists():
            env_example.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(
                f"  [{_OK}]\u2713[/] [{_FROST}].env.example[/] [{_STORM}]created[/]"
            )

    all_ok = all(ok for _, ok, _ in checks)
    console.print()
    if all_ok:
        console.print(f"  [bold {_OK}]Tramontane is ready.[/]\n")
    else:
        console.print(f"  [bold {_ERR}]Some checks failed. See errors above.[/]\n")


# ---------------------------------------------------------------------------
# tramontane hub (stub)
# ---------------------------------------------------------------------------


@app.command()
def hub(
    action: str = typer.Argument("search", help="search | install | publish"),
    name: Optional[str] = typer.Argument(None, help="Pipeline name (org/name)"),
) -> None:
    """Browse, install, and publish pipelines on Tramontane Hub."""
    console.print(
        Panel(
            f"[bold {_FROST}]Hub coming in v0.2[/]\n"
            f"[{_STORM}]Track at github.com/bleucommerce/tramontane[/]",
            border_style=f"dim {_RIM}",
            padding=(0, 2),
        )
    )


# ---------------------------------------------------------------------------
# tramontane serve (stub for Phase 5)
# ---------------------------------------------------------------------------


@app.command()
def serve(
    port: int = typer.Option(8080, help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    multitenancy: bool = typer.Option(False, help="Enable multitenancy"),
) -> None:
    """Start the Tramontane API server."""
    import uvicorn

    from tramontane.server.app import create_app

    _show_banner()
    console.print(
        f"  [{_CYAN}]Starting server on {host}:{port}[/]\n"
        f"  [{_STORM}]Docs at http://{host}:{port}/docs[/]\n"
    )
    fastapi_app = create_app(multitenancy=multitenancy)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
