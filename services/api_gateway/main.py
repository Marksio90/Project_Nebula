"""
services/api_gateway/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI HTTP entry-point for Project Nebula.

Endpoints:
  POST /mixes/generate   — Enqueue a full autonomous mix pipeline
  GET  /mixes/{mix_id}   — Poll mix status
  GET  /health           — Docker healthcheck probe
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.config import get_settings
from shared.tasks.celery_app import celery_app

log = logging.getLogger("nebula.api_gateway")
settings = get_settings()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API Gateway starting — environment=%s", settings.environment)
    yield
    log.info("API Gateway shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Project Nebula — AI Music Label API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateMixRequest(BaseModel):
    """Request body for autonomous mix generation."""
    requested_duration_minutes: int = Field(
        default=45,
        ge=10,
        le=120,
        description="Desired mix duration in minutes (10–120).",
    )
    style_hint: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Optional style override, e.g. 'Dark Neurofunk transitioning to Liquid'. "
            "If omitted the CSO agent decides autonomously."
        ),
    )
    force_bpm: int | None = Field(
        default=None,
        ge=60,
        le=220,
        description="Optional explicit BPM. CSO uses database-driven selection when None.",
    )


class GenerateMixResponse(BaseModel):
    mix_id: UUID
    task_id: str
    message: str


class MixStatusResponse(BaseModel):
    mix_id: UUID
    task_id: str
    state: str
    info: dict | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/mixes/generate",
    response_model=GenerateMixResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger autonomous mix generation pipeline",
)
async def generate_mix(request: GenerateMixRequest) -> GenerateMixResponse:
    mix_id = uuid4()
    # Import here to avoid circular imports at module load
    from shared.tasks.definitions import orchestrate_mix_pipeline

    task = orchestrate_mix_pipeline.apply_async(
        kwargs={
            "mix_id": str(mix_id),
            "requested_duration_minutes": request.requested_duration_minutes,
            "style_hint": request.style_hint,
            "force_bpm": request.force_bpm,
        },
        queue="orchestration",
    )
    log.info("Mix pipeline enqueued: mix_id=%s task_id=%s", mix_id, task.id)
    return GenerateMixResponse(
        mix_id=mix_id,
        task_id=task.id,
        message="Mix generation pipeline enqueued successfully.",
    )


@app.get(
    "/mixes/{mix_id}",
    response_model=MixStatusResponse,
    summary="Poll mix status by mix_id",
)
async def get_mix_status(mix_id: UUID) -> MixStatusResponse:
    # In production: look up task_id from the DB using mix_id.
    # For now, we accept task_id as the same as mix_id for simplicity.
    result = celery_app.AsyncResult(str(mix_id))
    if result is None:
        raise HTTPException(status_code=404, detail="Mix not found.")
    return MixStatusResponse(
        mix_id=mix_id,
        task_id=result.id,
        state=result.state,
        info=result.info if isinstance(result.info, dict) else None,
    )


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok", "service": "api_gateway"}
