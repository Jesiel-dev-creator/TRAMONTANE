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
    table.add_column("Reason", justify="center")
    table.add_column("Vision", justify="center")

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

        reason_str = (
            f"[{_OK}]\u2713[/]" if m.supports_reasoning_effort
            else f"[dim {_STORM}]\u2014[/]"
        )
        vision_str = (
            f"[{_OK}]\u2713[/]" if m.supports_vision
            else f"[dim {_STORM}]\u2014[/]"
        )

        table.add_row(
            name_str,
            tier_str,
            ", ".join(m.strengths[:3]),
            f"\u20ac{m.cost_per_1m_input_eur:.2f}",
            f"\u20ac{m.cost_per_1m_output_eur:.2f}",
            local_str,
            m.modality,
            reason_str,
            vision_str,
        )

    console.print(table)
    console.print(
        f"\n  [{_STORM}]Prices in EUR \u00b7 Source: Mistral AI \u00b7 Updated March 2026[/]\n"
    )


# ---------------------------------------------------------------------------
# tramontane doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Health check: verify API key, connectivity, models, telemetry."""
    import os

    from tramontane.router.models import MISTRAL_MODELS

    console.print()
    console.print(Rule(title="TRAMONTANE DOCTOR", style=f"bold {_CYAN}"))
    console.print()

    checks: list[tuple[str, bool, str]] = []

    # Version
    checks.append(("Version", True, tramontane.__version__))

    # API key
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:]
        checks.append(("MISTRAL_API_KEY", True, masked))
    else:
        checks.append(("MISTRAL_API_KEY", False, "not set"))

    # Models
    total = len(MISTRAL_MODELS)
    available = sum(1 for m in MISTRAL_MODELS.values() if m.available)
    checks.append(("Model fleet", True, f"{available}/{total} available"))

    # Telemetry
    try:
        from tramontane.router.telemetry import FleetTelemetry

        t = FleetTelemetry()
        checks.append(("Telemetry DB", True, f"{t.total_outcomes} outcomes recorded"))
    except Exception:
        checks.append(("Telemetry DB", True, "not initialized (OK)"))

    # Connectivity (quick, non-blocking)
    if api_key:
        try:
            from mistralai.client import Mistral

            client = Mistral(api_key=api_key)
            resp = asyncio.run(
                client.models.list_async()
            )
            count = len(resp.data) if hasattr(resp, "data") and resp.data else 0
            checks.append(("Mistral API", True, f"connected ({count} models)"))
        except Exception as exc:
            checks.append(("Mistral API", False, str(exc)[:60]))
    else:
        checks.append(("Mistral API", False, "skipped (no API key)"))

    for label, ok, detail in checks:
        icon = f"[{_OK}]\u2713[/]" if ok else f"[{_ERR}]\u2717[/]"
        console.print(f"  {icon} [{_FROST}]{label}[/] [{_STORM}]{detail}[/]")

    console.print()


# ---------------------------------------------------------------------------
# tramontane fleet
# ---------------------------------------------------------------------------


@app.command()
def fleet() -> None:
    """Show fleet status with per-model telemetry stats."""
    from tramontane.router.models import MISTRAL_MODELS

    console.print()
    console.print(Rule(title="FLEET STATUS", style=f"bold {_CYAN}"))
    console.print()

    table = Table(
        box=box.MINIMAL_HEAVY_HEAD,
        header_style=f"bold {_CYAN}",
        border_style=f"dim {_RIM}",
    )
    table.add_column("Model", style=_FROST)
    table.add_column("Tier", justify="center")
    table.add_column("EUR/1M in", justify="right", style=f"bold {_WARN}")
    table.add_column("EUR/1M out", justify="right", style=f"bold {_WARN}")
    table.add_column("Ctx", justify="right")
    table.add_column("Reason", justify="center")
    table.add_column("Vision", justify="center")
    table.add_column("Calls", justify="right")
    table.add_column("Success", justify="right")

    # Try to load telemetry stats
    model_stats: dict[str, dict[str, object]] = {}
    try:
        from tramontane.router.telemetry import FleetTelemetry

        t = FleetTelemetry()
        for stat in t.get_model_stats():
            model_stats[str(stat["model_used"])] = stat
    except Exception:
        pass

    for alias, m in MISTRAL_MODELS.items():
        reason_str = f"[{_OK}]\u2713[/]" if m.supports_reasoning_effort else "\u2014"
        vision_str = f"[{_OK}]\u2713[/]" if m.supports_vision else "\u2014"
        ctx_k = f"{m.context_window // 1000}K"

        stats = model_stats.get(alias, {})
        calls = str(stats.get("total", "\u2014"))
        total_val = stats.get("total", 0)
        total = int(total_val) if isinstance(total_val, (int, float)) else 0
        succ_val = stats.get("successes", 0)
        successes = int(succ_val) if isinstance(succ_val, (int, float)) else 0
        rate = f"{successes / total * 100:.0f}%" if total else "\u2014"

        table.add_row(
            alias,
            str(m.tier),
            f"\u20ac{m.cost_per_1m_input_eur:.2f}",
            f"\u20ac{m.cost_per_1m_output_eur:.2f}",
            ctx_k,
            reason_str,
            vision_str,
            calls,
            rate,
        )

    console.print(table)

    # Recommendations
    console.print(f"\n  [{_CYAN}]Recommended for common tasks:[/]")
    console.print(f"  [{_STORM}]General/Reasoning:[/] [{_FROST}]mistral-small-4[/]")
    console.print(f"  [{_STORM}]Code:[/]             [{_FROST}]devstral-small / devstral-2[/]")
    console.print(f"  [{_STORM}]Classification:[/]   [{_FROST}]ministral-3b[/]")
    console.print(f"  [{_STORM}]Frontier:[/]         [{_FROST}]mistral-large-3[/]")
    console.print()


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


# ---------------------------------------------------------------------------
# tramontane simulate
# ---------------------------------------------------------------------------


@app.command()
def simulate(
    pipeline_path: str = typer.Argument(..., help="Path to pipeline YAML"),
    input_text: str = typer.Option("sample input", "--input", "-i"),
) -> None:
    """Estimate pipeline cost without calling any API."""
    from tramontane.core.simulate import simulate_pipeline
    from tramontane.core.yaml_pipeline import create_agents_from_spec, load_pipeline_spec

    try:
        spec = load_pipeline_spec(pipeline_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"  [{_ERR}]{exc}[/]")
        raise typer.Exit(1) from exc

    agents = create_agents_from_spec(spec)
    sim = simulate_pipeline(agents, input_text, budget_eur=spec.budget_eur)

    console.print()
    console.print(Rule(title="COST SIMULATION", style=f"bold {_CYAN}"))

    table = Table(box=box.MINIMAL_HEAVY_HEAD, header_style=f"bold {_CYAN}")
    table.add_column("Agent", style=_FROST)
    table.add_column("Model", style=_CYAN)
    table.add_column("Est. Cost", justify="right", style=f"bold {_WARN}")
    table.add_column("Est. Time", justify="right")
    for a in sim.agents:
        table.add_row(
            a.role, a.model_predicted,
            f"\u20ac{a.estimated_cost_eur:.4f}", f"{a.estimated_time_s:.1f}s",
        )
    console.print(table)

    status_color = _OK if sim.budget_status == "within_budget" else _ERR
    console.print(
        f"\n  [{_FROST}]Total:[/] [{_WARN}]\u20ac{sim.total_estimated_cost_eur:.4f}[/]"
        f"  [{_FROST}]Time:[/] {sim.total_estimated_time_s:.1f}s"
        f"  [{status_color}]{sim.budget_status}[/]\n",
    )


# ---------------------------------------------------------------------------
# tramontane knowledge
# ---------------------------------------------------------------------------

knowledge_app = typer.Typer(help="Manage knowledge bases (RAG)")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("ingest")
def knowledge_ingest(
    path: str = typer.Argument(..., help="File path or glob pattern"),
    db: str = typer.Option("tramontane_knowledge.db", "--db"),
) -> None:
    """Ingest files into a knowledge base."""
    from tramontane.knowledge.base import KnowledgeBase

    kb = KnowledgeBase(db_path=db)
    count = asyncio.run(kb.ingest(sources=[path]))
    console.print(
        f"  [{_OK}]\u2713[/] [{_FROST}]Ingested {count} chunks[/]"
        f" [{_STORM}]into {db}[/]  [{_STORM}]Total: {kb.chunk_count}[/]",
    )


@knowledge_app.command("search")
def knowledge_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    db: str = typer.Option("tramontane_knowledge.db", "--db"),
) -> None:
    """Search the knowledge base."""
    from tramontane.knowledge.base import KnowledgeBase

    kb = KnowledgeBase(db_path=db)
    if kb.chunk_count == 0:
        console.print(
            f"  [{_WARN}]Knowledge base is empty."
            " Run 'tramontane knowledge ingest' first.[/]",
        )
        raise typer.Exit(1)

    result = asyncio.run(kb.retrieve(query, top_k=top_k))
    console.print()
    console.print(Rule(title="KNOWLEDGE SEARCH", style=f"bold {_CYAN}"))
    for chunk, score in zip(result.chunks, result.scores):
        console.print(
            f"  [{_CYAN}]{score:.2f}[/] [{_STORM}]{chunk.source}[/]",
        )
        console.print(f"  [{_FROST}]{chunk.content[:200]}...[/]\n")


# ---------------------------------------------------------------------------
# tramontane telemetry
# ---------------------------------------------------------------------------

telemetry_app = typer.Typer(help="Fleet telemetry and performance insights")
app.add_typer(telemetry_app, name="telemetry")


@telemetry_app.command("stats")
def telemetry_stats(
    db: str = typer.Option("tramontane_telemetry.db", "--db"),
) -> None:
    """Show fleet telemetry stats."""
    from tramontane.router.telemetry import FleetTelemetry

    t = FleetTelemetry(db_path=db)
    if t.total_outcomes == 0:
        console.print(f"  [{_WARN}]No telemetry data yet. Run some agents first.[/]")
        raise typer.Exit(0)

    stats = t.get_model_stats()
    console.print()
    console.print(Rule(title="FLEET TELEMETRY", style=f"bold {_CYAN}"))

    table = Table(box=box.MINIMAL_HEAVY_HEAD, header_style=f"bold {_CYAN}")
    table.add_column("Model", style=_FROST)
    table.add_column("Effort", style=_STORM)
    table.add_column("Calls", justify="right")
    table.add_column("Success", justify="right", style=f"bold {_OK}")
    table.add_column("Avg Cost", justify="right", style=f"bold {_WARN}")
    table.add_column("Avg Latency", justify="right")

    for s in stats:
        raw_total = s.get("total", 0)
        total = int(raw_total) if isinstance(raw_total, (int, float)) else 0
        raw_succ = s.get("successes", 0)
        succ = int(raw_succ) if isinstance(raw_succ, (int, float)) else 0
        raw_cost = s.get("avg_cost", 0)
        avg_cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else 0.0
        raw_lat = s.get("avg_latency", 0)
        avg_lat = float(raw_lat) if isinstance(raw_lat, (int, float)) else 0.0
        rate = f"{succ / total * 100:.0f}%" if total else "N/A"
        table.add_row(
            str(s.get("model_used", "")),
            str(s.get("reasoning_effort", "\u2014")),
            str(total), rate,
            f"\u20ac{avg_cost:.4f}", f"{avg_lat:.1f}s",
        )
    console.print(table)
    console.print(f"\n  [{_STORM}]Total outcomes: {t.total_outcomes}[/]\n")


# ---------------------------------------------------------------------------
# tramontane serve
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
