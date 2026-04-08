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

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from shared.config import get_settings
from shared.db.models import Mix, MixStatus, Stem, StemStatus, Visual, VisualStatus
from shared.db.session import get_db
from shared.tasks.celery_app import celery_app

# ── Step descriptions for live monitoring ─────────────────────────────────────
_STEP_DETAILS: dict[str, dict] = {
    "pending":          {"step": 0,  "label": "Oczekuje w kolejce",           "desc": "Zadanie zostało przyjęte i czeka na wolny worker."},
    "strategising":     {"step": 1,  "label": "CSO: Strategia AI",            "desc": "AI decyduje o BPM, subgatunku, kluczu i łuku narracyjnym."},
    "prompt_gen":       {"step": 2,  "label": "Generowanie promptów audio",   "desc": "AI tworzy szczegółowe prompty dla każdego 30-sekundowego stemu MusicGen."},
    "fetching_stems":   {"step": 3,  "label": "Generowanie stemów (Replicate)","desc": "Replicate MusicGen generuje każdy stem audio (~$0.008/stem)."},
    "stitching":        {"step": 4,  "label": "DSP: Beat matching + stitch",  "desc": "librosa wyrównuje beaty, crossfade na downbeat, łączy stemy."},
    "mastering":        {"step": 5,  "label": "DSP: Mastering",               "desc": "Pedalboard: EQ → kompresja → limiter → normalizacja LUFS -14."},
    "qa_audio":         {"step": 6,  "label": "QA Audio",                     "desc": "Weryfikacja LUFS ±0.5, true peak ≤ -1 dBFS."},
    "fetching_visuals": {"step": 7,  "label": "Generowanie wizuali (DALL-E 3)","desc": "DALL-E 3 generuje tła i okładki (~$0.08/obraz HD)."},
    "rendering":        {"step": 8,  "label": "Render video FFmpeg",          "desc": "FFmpeg łączy audio z wizualizacją spektrum i overlayami."},
    "slicing":          {"step": 9,  "label": "Cięcie Shortów",               "desc": "Analiza RMS → wybór 3 najgłośniejszych chwil → eksport 9:16."},
    "qa_video":         {"step": 10, "label": "QA Video",                     "desc": "Ffprobe sprawdza frame drops i sync audio/video."},
    "uploading":        {"step": 11, "label": "Upload YouTube + TikTok",      "desc": "Upload pełnego miksu + 3 Shortów z polskim SEO."},
    "complete":         {"step": 12, "label": "Gotowe! ✓",                    "desc": "Mix opublikowany na YouTube i TikTok."},
    "failed":           {"step": -1, "label": "Błąd",                         "desc": "Pipeline zatrzymany. Sprawdź error_message."},
}

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

# ── Static frontend ───────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
_FRONTEND_INDEX = os.path.join(_FRONTEND_DIR, "index.html")

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateMixRequest(BaseModel):
    """Request body for autonomous mix generation.

    The user provides only the genre — every other parameter (BPM, duration,
    key, subgenre, arc) is decided autonomously by the AI agents.
    """
    genre: str = Field(
        description="Music genre selected by the user. Must be a value from GET /genres.",
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


class MixProgressDetail(BaseModel):
    """Detailed progress for a single mix — stem & visual generation counts."""
    step_index:   int
    step_label:   str
    step_desc:    str
    stem_total:   int = 0
    stem_ready:   int = 0
    stem_failed:  int = 0
    visual_total: int = 0
    visual_ready: int = 0
    visual_failed: int = 0


class MixListItem(BaseModel):
    mix_id: UUID
    status: str
    bpm: float | None
    subgenre: str | None
    key_signature: str | None
    style_hint: str | None
    requested_duration_minutes: int
    actual_duration_seconds: float | None
    qa_passed: bool
    error_message: str | None
    created_at: str
    updated_at: str
    progress: MixProgressDetail | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/genres",
    response_model=list[str],
    summary="List all available music genres",
)
async def list_genres() -> list[str]:
    """Returns all 120+ selectable genre names for the frontend dropdown."""
    from shared.genres import GENRE_NAMES
    return GENRE_NAMES


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend() -> HTMLResponse:
    """Serve the MVP frontend dashboard."""
    if os.path.isfile(_FRONTEND_INDEX):
        with open(_FRONTEND_INDEX, encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read())
    return HTMLResponse(
        content="<h1>Project Nebula</h1><p>Frontend not found. See /docs for the API.</p>"
    )


@app.get(
    "/mixes",
    response_model=list[MixListItem],
    summary="List all mixes (most recent first)",
)
async def list_mixes(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[MixListItem]:
    from sqlalchemy import desc

    _TERMINAL = {MixStatus.COMPLETE, MixStatus.FAILED}

    async with get_db() as session:
        result = await session.execute(
            select(Mix).order_by(desc(Mix.created_at)).offset(offset).limit(limit)
        )
        mixes = result.scalars().all()

        # Fetch stem + visual counts for non-terminal mixes (only active ones need live data)
        active_ids = [m.id for m in mixes if m.status not in _TERMINAL]

        stem_map: dict[str, dict] = {}
        visual_map: dict[str, dict] = {}

        if active_ids:
            # Stem counts per mix — using PostgreSQL FILTER aggregate
            stem_rows = await session.execute(
                select(
                    Stem.mix_id,
                    func.count(Stem.id).label("total"),
                    func.count(Stem.id).filter(Stem.status == StemStatus.READY.value).label("s_ready"),
                    func.count(Stem.id).filter(Stem.status == StemStatus.FAILED.value).label("s_failed"),
                )
                .where(Stem.mix_id.in_(active_ids))
                .group_by(Stem.mix_id)
            )
            for row in stem_rows.mappings().all():
                stem_map[row["mix_id"]] = {
                    "total":  row["total"]   or 0,
                    "ready":  row["s_ready"] or 0,
                    "failed": row["s_failed"] or 0,
                }

            # Visual counts per mix
            visual_rows = await session.execute(
                select(
                    Visual.mix_id,
                    func.count(Visual.id).label("total"),
                    func.count(Visual.id).filter(Visual.status == VisualStatus.READY.value).label("v_ready"),
                    func.count(Visual.id).filter(Visual.status == VisualStatus.FAILED.value).label("v_failed"),
                )
                .where(Visual.mix_id.in_(active_ids))
                .group_by(Visual.mix_id)
            )
            for row in visual_rows.mappings().all():
                visual_map[row["mix_id"]] = {
                    "total":  row["total"]   or 0,
                    "ready":  row["v_ready"] or 0,
                    "failed": row["v_failed"] or 0,
                }

    items = []
    for m in mixes:
        step_info = _STEP_DETAILS.get(m.status.value, _STEP_DETAILS["pending"])
        sc = stem_map.get(m.id, {})
        vc = visual_map.get(m.id, {})
        progress = MixProgressDetail(
            step_index=step_info["step"],
            step_label=step_info["label"],
            step_desc=step_info["desc"],
            stem_total=sc.get("total", 0),
            stem_ready=sc.get("ready", 0),
            stem_failed=sc.get("failed", 0),
            visual_total=vc.get("total", 0),
            visual_ready=vc.get("ready", 0),
            visual_failed=vc.get("failed", 0),
        ) if m.status not in _TERMINAL else None
        items.append(MixListItem(
            mix_id=UUID(m.id),
            status=m.status.value,
            bpm=m.bpm,
            subgenre=m.subgenre,
            key_signature=m.key_signature,
            style_hint=m.style_hint,
            requested_duration_minutes=m.requested_duration_minutes,
            actual_duration_seconds=m.actual_duration_seconds,
            qa_passed=m.qa_passed,
            error_message=m.error_message,
            created_at=m.created_at.isoformat(),
            updated_at=m.updated_at.isoformat(),
            progress=progress,
        ))
    return items


@app.post(
    "/mixes/generate",
    response_model=GenerateMixResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger autonomous mix generation pipeline",
)
async def generate_mix(request: GenerateMixRequest) -> GenerateMixResponse:
    from shared.genres import GENRE_NAMES
    if request.genre not in GENRE_NAMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown genre '{request.genre}'. Call GET /genres for the full list.",
        )

    mix_id = uuid4()
    from shared.tasks.definitions import orchestrate_mix_pipeline

    task = await run_in_threadpool(
        lambda: orchestrate_mix_pipeline.apply_async(
            kwargs={
                "mix_id": str(mix_id),
                "genre": request.genre,
            },
            queue="orchestration",
        )
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
        celery_state = await run_in_threadpool(
            lambda: celery_app.AsyncResult(mix.celery_task_id).state
        )

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
