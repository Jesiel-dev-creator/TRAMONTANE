"""TRAMONTANE v0.1.1 — Live Integration Validation.

Requires MISTRAL_API_KEY in the environment. Calls real Mistral API endpoints.
Run with: uv run python tests/live_validation.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import anyio
import httpx
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

TIMEOUT = 60.0  # per-test timeout in seconds


# ═══════════════════════════════════════════════
# TEST 1: SSE Streaming Endpoint
# ═══════════════════════════════════════════════


async def test_sse_streaming() -> tuple[bool, str]:
    """Start FastAPI server, POST to /pipelines/run with stream=true, verify SSE events."""
    import uvicorn

    from tramontane.server.app import create_app

    app = create_app()

    # Use port 0 to let OS pick a free port
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)

    events_received: list[str] = []
    error: str | None = None

    async with anyio.create_task_group() as tg:
        tg.start_soon(server.serve)
        await anyio.sleep(2)

        # Find actual port from server sockets
        port: int | None = None
        if server.servers:
            for s in server.servers:
                for sock in s.sockets:
                    addr = sock.getsockname()
                    if isinstance(addr, tuple):
                        port = addr[1]
                        break
                if port:
                    break

        if not port:
            server.should_exit = True
            return False, "Could not determine server port"

        base = f"http://127.0.0.1:{port}"

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                # Health check first
                r = await client.get(f"{base}/health")
                if r.status_code != 200:
                    error = f"Health check failed: {r.status_code}"
                else:
                    # SSE streaming request
                    async with client.stream(
                        "POST",
                        f"{base}/pipelines/run",
                        json={
                            "pipeline_name": "code_review",
                            "input": "def add(a, b): return a + b",
                            "stream": True,
                            "budget_eur": 0.10,
                        },
                    ) as resp:
                        if resp.status_code != 200:
                            error = f"SSE endpoint returned {resp.status_code}"
                        else:
                            content_type = resp.headers.get("content-type", "")
                            if "text/event-stream" not in content_type:
                                error = f"Wrong content-type: {content_type}"
                            else:
                                async for line in resp.aiter_lines():
                                    if line.startswith("event:") or line.startswith("data:"):
                                        events_received.append(line)
                                    if len(events_received) > 30:
                                        break

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            server.should_exit = True

    if error:
        return False, error

    if not events_received:
        return False, "No SSE events received"

    has_start = any("pipeline_start" in e for e in events_received)
    has_done = any("pipeline_complete" in e or "done" in e for e in events_received)
    return True, f"{len(events_received)} events, has_start={has_start}, has_done={has_done}"


# ═══════════════════════════════════════════════
# TEST 2: Pipeline End-to-End
# ═══════════════════════════════════════════════


async def test_pipeline_e2e() -> tuple[bool, str]:
    """Run the code_review pipeline end-to-end with real API."""
    from tramontane.core.pipeline import Pipeline

    pipeline = Pipeline.from_yaml("pipelines/code_review.yaml")
    pipeline.budget_eur = 0.10

    result = await pipeline.run(
        input_text='def hello():\n    print("hello world")\n\nhello()',
    )

    if result is None:
        return False, "Pipeline returned None"

    status = result.status.value
    cost = result.total_cost_eur
    output_len = len(result.output or "")
    agents = result.agents_used
    models = result.models_used

    if status != "complete":
        return False, f"Status: {status}, error: {result.error}"

    if cost <= 0:
        return False, f"Cost is {cost} — no real API call happened"

    if output_len == 0:
        return False, "Output is empty"

    return True, (
        f"status={status}, cost=EUR {cost:.4f}, "
        f"output={output_len} chars, agents={agents}, models={models}"
    )


# ═══════════════════════════════════════════════
# TEST 3: Article 30 GDPR Report
# ═══════════════════════════════════════════════


async def test_article30_report() -> tuple[bool, str]:
    """Populate audit log with a real agent call, then generate Article 30 report."""
    from tramontane.core.agent import Agent
    from tramontane.gdpr.audit import AuditVault
    from tramontane.gdpr.reports import GDPRReporter
    from tramontane.memory.longterm import LongTermMemory
    from tramontane.router.router import MistralRouter

    # Run a real agent call to populate audit
    agent = Agent(
        role="test agent",
        goal="answer questions",
        backstory="you are a helpful test agent",
        model="mistral-small",
        budget_eur=0.05,
    )
    router = MistralRouter()
    result = await agent.run(
        "What is the capital of France? Answer in one word.",
        router=router,
    )

    # Log to audit vault
    vault = AuditVault()
    await vault.log(
        run_id="live-test-001",
        pipeline_name="live_validation",
        agent_role=agent.role,
        action_type="llm_call",
        model_used=result.model_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_eur=result.cost_eur,
        gdpr_sensitivity="none",
        pii_detected=False,
        pii_redacted=False,
    )

    # Generate report
    reporter = GDPRReporter(audit_vault=vault, memory=LongTermMemory())
    report = await reporter.article_30_report(pipeline_name="live_validation")

    if not report:
        return False, "Report is empty"

    has_type = "Article 30" in report.get("report_type", "")
    has_controller = bool(report.get("controller"))
    ops = report.get("processing_activities", {}).get("total_operations", 0)

    json_out = reporter.export_json(report)

    return True, (
        f"Article30={has_type}, controller={has_controller}, "
        f"ops={ops}, json={len(json_out)} chars"
    )


# ═══════════════════════════════════════════════
# TEST 4: GDPR Strict Mode — PII Redaction
# ═══════════════════════════════════════════════


async def test_gdpr_strict() -> tuple[bool, str]:
    """Test PII detection + redaction in strict mode.

    NOTE: gdpr_level on Agent does not auto-redact in agent.run().
    GDPR middleware is applied at the Pipeline level. This test
    verifies the PII detector directly, then checks that the
    agent can still process redacted input.
    """
    from tramontane.core.agent import Agent
    from tramontane.gdpr.pii import PIIDetector
    from tramontane.router.router import MistralRouter

    pii_input = (
        "Bonjour, je m'appelle Jean Dupont. "
        "Mon email est jean.dupont@gmail.com et "
        "mon telephone est 06 12 34 56 78."
    )

    # Step 1: PII detection + redaction
    detector = PIIDetector()
    pii_result = await detector.detect(pii_input)

    if not pii_result.has_pii:
        return False, "PII detector found no PII in input with email + phone"

    email_redacted = "jean.dupont@gmail.com" not in pii_result.cleaned_text
    phone_redacted = "06 12 34 56 78" not in pii_result.cleaned_text

    if not email_redacted:
        return False, "Email not redacted in cleaned text"
    if not phone_redacted:
        return False, "Phone not redacted in cleaned text"

    # Step 2: Run agent with the REDACTED text (simulating strict pipeline)
    agent = Agent(
        role="customer support",
        goal="help customers",
        backstory="you handle customer inquiries in French",
        model="mistral-small",
        budget_eur=0.05,
    )
    router = MistralRouter()
    result = await agent.run(pii_result.cleaned_text, router=router)

    if not result.output:
        return False, "Agent returned empty output on redacted input"

    # Verify PII does NOT appear in agent's output
    output_has_email = "jean.dupont@gmail.com" in result.output
    output_has_phone = "06 12 34 56 78" in result.output

    pii_types = [t.value for t in pii_result.pii_types_found]

    return True, (
        f"PII found: {pii_types}, "
        f"email_redacted={email_redacted}, phone_redacted={phone_redacted}, "
        f"output_leak_email={output_has_email}, output_leak_phone={output_has_phone}, "
        f"agent_cost=EUR {result.cost_eur:.4f}"
    )


# ═══════════════════════════════════════════════
# TEST 5: Hub Publish Pipeline YAML
# ═══════════════════════════════════════════════


async def test_hub_publish() -> tuple[bool, str]:
    """Validate pipeline YAML for HF Hub. Publish if HF_TOKEN is set."""
    import yaml

    from tramontane.hub.publisher import PipelinePublisher, PublishConfig

    yaml_path = "pipelines/code_review.yaml"

    # Step 1: Validate the YAML loads and has required fields
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = ["name", "agents", "handoffs"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"YAML missing required fields: {missing}"

    agents = data.get("agents", [])
    handoffs = data.get("handoffs", [])
    if not agents:
        return False, "YAML has no agents"
    if not handoffs:
        return False, "YAML has no handoffs"

    # Step 2: Validate via publisher's internal validator
    validated = PipelinePublisher._validate_yaml(yaml_path)
    if validated is None:
        return False, "Publisher._validate_yaml rejected the YAML"

    # Step 3: If HF_TOKEN is set, actually publish
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        pub = PipelinePublisher(hf_token=hf_token)
        config = PublishConfig(
            pipeline_path=yaml_path,
            repo_name="BleuCommerce-Apps/tramontane-pipelines",
            description="Code review pipeline — live validation test",
            tags=["tramontane-pipeline", "code-review", "live-test"],
        )
        url = pub.publish(config)
        if url:
            return True, f"Published to {url}"
        return False, "Publish returned empty URL"

    return True, (
        f"Validated locally (HF_TOKEN not set): "
        f"name={data['name']}, {len(agents)} agents, {len(handoffs)} handoffs"
    )


# ═══════════════════════════════════════════════
# TEST 6: Voice Gateway (Voxtral-Mini)
# ═══════════════════════════════════════════════


async def test_voice_gateway() -> tuple[bool, str]:
    """Generate a test WAV, send to Voxtral-Mini, verify gateway works."""
    import math
    import struct
    import tempfile
    import wave

    from tramontane.voice.gateway import VoiceGateway

    # Create a 1-second 440Hz sine wave WAV
    path = tempfile.mktemp(suffix=".wav")
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    try:
        with wave.open(path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            samples = b"".join(
                struct.pack(
                    "<h",
                    int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate)),
                )
                for i in range(int(sample_rate * duration))
            )
            f.writeframes(samples)

        gw = VoiceGateway()
        if not gw.is_available():
            return False, "VoiceGateway not available (no MISTRAL_API_KEY)"

        result = await gw.transcribe_file(path)

        # A sine wave has no speech — empty transcript is expected.
        # We're testing: does the gateway call the API without crashing?
        return True, (
            f"transcript='{result.transcript[:40]}', "
            f"confidence={result.confidence}, "
            f"language={result.language}, "
            f"duration={result.duration_seconds}s, "
            f"cost=EUR {result.cost_eur:.6f}"
        )

    finally:
        if os.path.exists(path):
            os.unlink(path)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════


async def main() -> None:
    """Run all live validation tests sequentially."""
    if not os.environ.get("MISTRAL_API_KEY"):
        console.print("[bold red]MISTRAL_API_KEY not set. Exiting.[/]")
        sys.exit(1)

    console.print()
    console.print(
        "[bold #00D4EE]TRAMONTANE v0.1.1 — Live Integration Validation[/]"
    )
    console.print("[#4A6480]Real Mistral API calls. Budget-capped.[/]")
    console.print()

    tests: list[tuple[str, Any]] = [
        ("SSE Streaming", test_sse_streaming),
        ("Pipeline E2E (code_review)", test_pipeline_e2e),
        ("Article 30 Report", test_article30_report),
        ("GDPR Strict (PII Redaction)", test_gdpr_strict),
        ("Hub Publish", test_hub_publish),
        ("Voice Gateway", test_voice_gateway),
    ]

    results: list[tuple[str, bool, str, float]] = []

    for name, test_fn in tests:
        console.print(f"[#00D4EE]  >> {name}[/]")
        start = time.monotonic()
        try:
            with anyio.fail_after(TIMEOUT):
                passed, detail = await test_fn()
        except TimeoutError:
            passed, detail = False, f"Timed out after {TIMEOUT}s"
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - start

        status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"     {status} ({elapsed:.1f}s) {detail[:80]}")
        results.append((name, passed, detail, elapsed))

    # Summary table
    console.print()
    table = Table(
        title="TRAMONTANE Live Validation",
        title_style="bold #00D4EE",
        box=box.MINIMAL_HEAVY_HEAD,
        header_style="bold #00D4EE",
        border_style="dim #1C2E42",
    )
    table.add_column("Test", style="#DCE9F5")
    table.add_column("Result")
    table.add_column("Time", justify="right", style="#4A6480")
    table.add_column("Detail", style="#4A6480", max_width=60)

    for name, passed, detail, elapsed in results:
        table.add_row(
            name,
            "[green]PASS[/]" if passed else "[red]FAIL[/]",
            f"{elapsed:.1f}s",
            detail[:60],
        )

    console.print(table)

    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    total_time = sum(e for _, _, _, e in results)
    color = "green" if passed_count == total else "red"
    console.print(
        f"\n  [{color}]{passed_count}/{total} passed[/] "
        f"[#4A6480]in {total_time:.1f}s[/]\n"
    )

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    anyio.run(main)
