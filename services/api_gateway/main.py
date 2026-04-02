"""
services/api_gateway/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI HTTP entry-point for Project Nebula.

Endpoints:
  POST /mixes/generate              — Enqueue a full autonomous mix pipeline
  GET  /mixes/{mix_id}              — Poll mix status (source of truth: DB)
  GET  /mixes/{mix_id}/audio/download — Stream mastered audio WAV file
  GET  /health                      — Docker healthcheck probe
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from shared.config import get_settings
from shared.db.models import Mix
from shared.db.session import get_db
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
    async with get_db() as session:
        mix = await session.get(Mix, str(mix_id))

    if mix is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mix {mix_id} not found. It may still be queued.",
        )

    # Enrich with live Celery task state when available
    celery_state: str | None = None
    if mix.celery_task_id:
        celery_state = celery_app.AsyncResult(mix.celery_task_id).state

    return MixStatusResponse(
        mix_id=mix_id,
        task_id=mix.celery_task_id or "",
        state=mix.status.value,
        info={"celery_state": celery_state} if celery_state else None,
    )


@app.get(
    "/mixes/{mix_id}/audio/download",
    summary="Download mastered audio for a completed mix",
    response_class=FileResponse,
)
async def download_mix_audio(mix_id: UUID) -> FileResponse:
    async with get_db() as session:
        mix = await session.get(Mix, str(mix_id))

    if mix is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mix {mix_id} not found.",
        )

    if not mix.mastered_audio_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mix {mix_id} has no audio yet (status: {mix.status.value}).",
        )

    if not os.path.isfile(mix.mastered_audio_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found on disk. The mix may still be processing.",
        )

    filename = f"nebula_mix_{mix_id}.wav"
    return FileResponse(
        path=mix.mastered_audio_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok", "service": "api_gateway"}
