"""Tramontane API routes — REST + SSE streaming.

All endpoints follow clean REST design with proper HTTP status codes.
Pipeline runs support SSE streaming via text/event-stream.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import tramontane
from tramontane.router.models import MISTRAL_MODELS

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PipelineRunRequest(BaseModel):
    """Request body for POST /pipelines/run."""

    pipeline_yaml: str | None = None
    pipeline_name: str | None = None
    input: str
    budget_eur: float | None = None
    gdpr_level: str = "none"
    locale: str = "en"
    local_mode: bool = False
    stream: bool = True
    tenant_id: str | None = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    db: str
    timestamp: str


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check — always returns 200."""
    db_status = "ok"
    try:
        from tramontane.memory.longterm import LongTermMemory

        db_path: str = getattr(request.app.state, "db_path", "tramontane.db")
        mem = LongTermMemory(db_path=db_path)
        mem._get_db()
    except Exception:
        logger.warning("Health check: database error", exc_info=True)
        db_status = "error"

    return HealthResponse(
        status="ok",
        version=tramontane.__version__,
        db=db_status,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models() -> list[dict[str, Any]]:
    """List all Mistral models in the fleet."""
    result: list[dict[str, Any]] = []
    for alias, m in MISTRAL_MODELS.items():
        result.append({
            "alias": alias,
            "api_id": m.api_id,
            "tier": m.tier,
            "cost_input_eur": m.cost_per_1m_input_eur,
            "cost_output_eur": m.cost_per_1m_output_eur,
            "strengths": m.strengths,
            "local_available": m.local_ollama is not None,
            "hf_downloads": m.hf_downloads,
            "modality": m.modality,
            "available": m.available,
        })
    return result


@router.get("/models/{alias}")
async def get_model(alias: str) -> dict[str, Any]:
    """Get details for a single model."""
    from tramontane.router.models import get_model as _get_model

    m = _get_model(alias)
    return {
        "alias": alias,
        "api_id": m.api_id,
        "tier": m.tier,
        "cost_input_eur": m.cost_per_1m_input_eur,
        "cost_output_eur": m.cost_per_1m_output_eur,
        "strengths": m.strengths,
        "context_window": m.context_window,
        "local_ollama": m.local_ollama,
        "license": m.license,
        "modality": m.modality,
        "available": m.available,
    }


# ---------------------------------------------------------------------------
# /pipelines
# ---------------------------------------------------------------------------


@router.post("/pipelines/run")
async def run_pipeline(body: PipelineRunRequest) -> Any:
    """Run a pipeline. Returns SSE stream or JSON result."""
    from tramontane.core.pipeline import Pipeline, PipelineMode
    from tramontane.server.streaming import PipelineStreamer

    pipe: Pipeline | None = None

    if body.pipeline_yaml:
        import yaml as _yaml

        from tramontane.core.agent import Agent

        data: dict[str, Any] = _yaml.safe_load(body.pipeline_yaml)
        agents = [
            Agent(**{k: v for k, v in a.items() if k != "id"})
            for a in data.get("agents", [])
        ]
        handoffs: list[tuple[str, str]] = [
            (h["from"], h["to"]) if isinstance(h, dict) else (h[0], h[1])
            for h in data.get("handoffs", [])
        ]
        pipe = Pipeline(
            name=data.get("name", "api_pipeline"),
            agents=agents,
            handoffs=handoffs,
            budget_eur=body.budget_eur,
            mode=PipelineMode(data.get("mode", "agentic")),
            gdpr_level=body.gdpr_level,
            locale=body.locale,
        )
    elif body.pipeline_name:
        pipe = Pipeline.from_yaml(f"pipelines/{body.pipeline_name}.yaml")
        if body.budget_eur is not None:
            pipe.budget_eur = body.budget_eur
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "pipeline_yaml or pipeline_name required"},
        )

    if body.stream:
        run_id = uuid.uuid4().hex
        streamer = PipelineStreamer(pipe, run_id=run_id)
        return StreamingResponse(
            streamer.stream(body.input),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    pipe_run = await pipe.run(input_text=body.input)
    return pipe_run.model_dump(mode="json")


@router.get("/pipelines/{run_id}")
async def get_pipeline_run(run_id: str, request: Request) -> Any:
    """Get status and cost breakdown for a pipeline run."""
    import sqlite3

    db_path: str = getattr(request.app.state, "db_path", "tramontane.db")
    try:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        cursor = db.execute(
            "SELECT * FROM pipeline_checkpoints WHERE run_id = ? ORDER BY step",
            (run_id,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        db.close()
    except Exception:
        rows = []

    if not rows:
        return JSONResponse(status_code=404, content={"error": "run not found"})

    return {
        "run_id": run_id,
        "steps": rows,
        "total_cost_eur": sum(float(r.get("cost_eur", 0)) for r in rows),
    }


@router.get("/pipelines/{run_id}/audit")
async def get_pipeline_audit(run_id: str) -> list[dict[str, Any]]:
    """Get audit log for a pipeline run."""
    from tramontane.gdpr.audit import AuditVault

    vault = AuditVault()
    entries = await vault.get_run(run_id)
    return [e.model_dump(mode="json") for e in entries]


# ---------------------------------------------------------------------------
# /runs
# ---------------------------------------------------------------------------


@router.get("/runs")
async def list_runs(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent pipeline runs from the checkpoint database."""
    import sqlite3

    try:
        db = sqlite3.connect("tramontane.db")
        db.row_factory = sqlite3.Row
        cursor = db.execute(
            "SELECT DISTINCT run_id, agent_role, cost_eur, created_at "
            "FROM pipeline_checkpoints ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        db.close()
        return rows
    except Exception:
        return []


@router.delete("/runs/{run_id}")
async def archive_run(run_id: str) -> dict[str, str]:
    """Archive a run (no actual deletion — audit trail preserved)."""
    logger.info("Run %s marked as archived", run_id)
    return {"status": "archived", "run_id": run_id}
