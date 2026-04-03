"""
shared/tasks/definitions.py
─────────────────────────────────────────────────────────────────────────────
Core Celery task definitions for Project Nebula.

Pipeline topology (Celery chain + chord):

  orchestrate_mix_pipeline
    └─ chain:
        1. run_cso_strategy            [orchestration]  CSO → BPM/subgenre/arc
        2. generate_audio_prompts      [orchestration]  90+ Lyria 3 English prompts
        3. fetch_stems_from_gemini     [orchestration]  Parallel Lyria 3 API calls
        4. stitch_and_master_audio     [dsp]            librosa + pedalboard
        5. run_qa_audio_check          [dsp]            LUFS / true-peak verify
        6. generate_visual_prompts     [orchestration]  Nano Banana 2 / Veo prompts
        7. fetch_visuals_from_gemini   [orchestration]  Parallel visual API calls
        8. render_full_video           [video]          FFmpeg 16:9 full mix
        9. slice_viral_shorts          [video]          3× 60s 9:16 RMS clips
       10. run_qa_video_check          [video]          Frame-drop detection
       11. generate_polish_seo         [orchestration]  Polish metadata via CrewAI
       12. upload_to_youtube           [upload]         Full mix + Shorts
       13. upload_to_tiktok            [upload]         Viral Shorts

Retry strategy (tenacity + Celery native):
  - API-bound tasks: exponential backoff, max 5 retries, jitter
  - DSP/video tasks: fixed 30s delay, max 3 retries
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import time
from typing import Any

from celery import chain, chord, group, signature
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select, update

from shared.db.models import (
    BpmSubgenreRegistry,
    Mix,
    MixStatus,
    Platform,
    PlatformUpload,
    Stem,
    StemStatus,
    UploadStatus,
    ViralShort,
    Visual,
    VisualStatus,
    VisualType,
)
from shared.db.session import get_sync_db
from shared.schemas.events import (
    AudioPromptBatch,
    AudioQAResult,
    AudioStitchResult,
    CSOStrategy,
    MixPipelineRequest,
    PolishSEOMetadata,
    StemBatchResult,
    StemFetchResult,
    StemPrompt,
    UploadResult,
    VideoQAResult,
    VideoRenderResult,
    ViralShortResult,
    ViralSliceResult,
    VisualPrompt,
    VisualPromptBatch,
)
from shared.tasks.celery_app import celery_app

log = logging.getLogger("nebula.tasks")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _update_mix_status(mix_id: str, status: MixStatus, **extra_fields) -> None:
    """Convenience: update a Mix row's status + optional fields in one commit."""
    with get_sync_db() as db:
        values = {"status": status, **extra_fields}
        db.execute(update(Mix).where(Mix.id == mix_id).values(**values))


def _mark_mix_failed(mix_id: str, error: str) -> None:
    log.error("Mix %s FAILED: %s", mix_id, error)
    _update_mix_status(mix_id, MixStatus.FAILED, error_message=error[:2000])


_OPENAI_QUOTA_PHRASES = (
    "exceeded your current quota",
    "check your plan and billing",
)


def _is_quota_exhausted(exc: Exception) -> bool:
    """Return True when the OpenAI account has run out of billing credits.
    These errors are permanent — retrying wastes time and log noise."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _OPENAI_QUOTA_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0 — Pipeline Orchestrator (entry-point)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.orchestrate_mix_pipeline",
    bind=True,
    queue="orchestration",
    max_retries=0,   # Top-level orchestrator should not retry blindly
    acks_late=True,
)
def orchestrate_mix_pipeline(
    self,
    mix_id: str,
    genre: str = "Drum and Bass",
) -> dict:
    """
    Entry-point task.  Creates the Mix row and fires the full Celery chain.
    Duration is chosen autonomously based on genre profile + recent history.
    Returns immediately so the API gateway can respond 202 Accepted.
    """
    from sqlalchemy import desc as _desc
    from shared.genres import pick_intelligent_duration

    # Pick an intelligent duration: genre-appropriate + avoids recent repeats
    with get_sync_db() as db:
        recent_durations = db.execute(
            select(Mix.requested_duration_minutes)
            .where(Mix.status == MixStatus.COMPLETE)
            .order_by(_desc(Mix.created_at))
            .limit(20)
        ).scalars().all()

    requested_duration_minutes = pick_intelligent_duration(genre, list(recent_durations))
    log.info(
        "▶ orchestrate_mix_pipeline mix_id=%s genre=%s duration=%dmin",
        mix_id, genre, requested_duration_minutes,
    )

    # Persist the Mix record
    with get_sync_db() as db:
        mix = Mix(
            id=str(mix_id),
            celery_task_id=self.request.id,
            status=MixStatus.PENDING,
            requested_duration_minutes=requested_duration_minutes,
            style_hint=genre,   # Store genre in style_hint for display/history
        )
        db.add(mix)

    request_payload = MixPipelineRequest(
        mix_id=str(mix_id),
        genre=genre,
        requested_duration_minutes=requested_duration_minutes,
    ).model_dump()

    # Build the sequential pipeline chain
    pipeline = chain(
        run_cso_strategy.si(request_payload),
        generate_audio_prompts.s(),
        fetch_stems_from_gemini.s(),
        stitch_and_master_audio.s(),
        run_qa_audio_check.s(),
        generate_visual_prompts.s(),
        fetch_visuals_from_gemini.s(),
        render_full_video.s(),
        slice_viral_shorts.s(),
        run_qa_video_check.s(),
        generate_polish_seo.s(),
        upload_to_youtube.s(),
        upload_to_tiktok.s(),
    )
    pipeline.apply_async()

    return {"mix_id": mix_id, "status": "pipeline_started"}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — CSO Strategy Agent
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.run_cso_strategy",
    bind=True,
    queue="orchestration",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def run_cso_strategy(self, request_payload: dict) -> dict:
    """
    Chief Strategy Officer agent.
    Queries the BPM/subgenre registry to select a genuinely novel combination,
    then delegates to CrewAI for the full strategic brief.
    Returns a serialised CSOStrategy dict.
    """
    req = MixPipelineRequest(**request_payload)
    log.info("🎯 CSO strategy: mix_id=%s genre=%s duration=%dmin",
             req.mix_id, req.genre, req.requested_duration_minutes)
    _update_mix_status(req.mix_id, MixStatus.STRATEGISING)

    try:
        from services.orchestrator.crew.agents import run_cso_agent

        strategy: CSOStrategy = run_cso_agent(
            mix_id=req.mix_id,
            genre=req.genre,
            requested_duration_minutes=req.requested_duration_minutes,
        )
    except Exception as exc:
        log.warning("CSO strategy error (attempt %d): %s", self.request.retries + 1, exc)
        if _is_quota_exhausted(exc):
            _mark_mix_failed(req.mix_id, f"OpenAI quota exhausted — add credits: {exc}")
            raise
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 15)
        except MaxRetriesExceededError:
            _mark_mix_failed(req.mix_id, f"CSO strategy failed after retries: {exc}")
            raise

    # Persist dedup registry entry
    with get_sync_db() as db:
        registry_entry = BpmSubgenreRegistry(
            mix_id=strategy.mix_id,
            bpm=strategy.bpm,
            subgenre=strategy.subgenre,
            key_signature=strategy.key_signature,
        )
        db.merge(registry_entry)
        db.execute(
            update(Mix).where(Mix.id == req.mix_id).values(
                bpm=strategy.bpm,
                subgenre=strategy.subgenre,
                key_signature=strategy.key_signature,
                stem_count=strategy.stem_count,
                status=MixStatus.PROMPT_GEN,
            )
        )

    log.info("✅ CSO: bpm=%.1f subgenre=%s stems=%d", strategy.bpm, strategy.subgenre, strategy.stem_count)
    return strategy.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Audio Prompt Engineer
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.generate_audio_prompts",
    bind=True,
    queue="orchestration",
    max_retries=3,
    default_retry_delay=20,
    acks_late=True,
)
def generate_audio_prompts(self, strategy_dict: dict) -> dict:
    """
    Audio Prompt Engineer agent.
    Translates the CSO's CSOStrategy into N individual StemPrompts
    (one per 30-second stem), ensuring a coherent musical arc.
    All prompts are in English for Gemini Lyria 3.
    """
    strategy = CSOStrategy(**strategy_dict)
    log.info("🎵 Audio prompts: mix_id=%s stems=%d", strategy.mix_id, strategy.stem_count)

    try:
        from services.orchestrator.crew.agents import run_audio_prompt_engineer

        batch: AudioPromptBatch = run_audio_prompt_engineer(strategy=strategy)
    except Exception as exc:
        log.warning("Audio prompt error: %s", exc)
        if _is_quota_exhausted(exc):
            _mark_mix_failed(strategy.mix_id, f"OpenAI quota exhausted — add credits: {exc}")
            raise
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 10)
        except MaxRetriesExceededError:
            _mark_mix_failed(strategy.mix_id, f"Audio prompt generation failed: {exc}")
            raise

    # Persist Stem rows (status=PENDING)
    with get_sync_db() as db:
        for p in batch.prompts:
            stem = Stem(
                mix_id=strategy.mix_id,
                position=p.position,
                gemini_prompt=p.prompt_en,
                status=StemStatus.PENDING,
            )
            db.add(stem)

    log.info("✅ Audio prompts generated: %d stems", len(batch.prompts))
    return batch.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Gemini Stem Fetcher
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.fetch_stems_from_gemini",
    bind=True,
    queue="orchestration",
    max_retries=5,
    acks_late=True,
    soft_time_limit=3600,    # 1-hour soft limit for batch generation
    time_limit=3900,
)
def fetch_stems_from_gemini(self, batch_dict: dict) -> dict:
    """
    Calls Gemini Lyria 3 for each StemPrompt.
    Uses tenacity for per-stem retry with exponential backoff + jitter
    to gracefully absorb API rate limits.
    Stems are downloaded to STEMS_DIR/{mix_id}/{position:04d}.wav
    """
    batch = AudioPromptBatch(**batch_dict)
    mix_id = batch.strategy.mix_id
    log.info("🌐 Fetching %d stems from Gemini Lyria 3: mix_id=%s", len(batch.prompts), mix_id)
    _update_mix_status(mix_id, MixStatus.FETCHING_STEMS)

    try:
        from services.orchestrator.crew.agents import fetch_stems_batch

        result: StemBatchResult = fetch_stems_batch(batch=batch)
    except Exception as exc:
        log.warning("Stem fetch error: %s", exc)
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 30)
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Stem fetching failed after retries: {exc}")
            raise

    # Persist stem file paths and statuses
    with get_sync_db() as db:
        for r in result.results:
            db.execute(
                update(Stem)
                .where(Stem.mix_id == mix_id, Stem.position == r.position)
                .values(
                    file_path=r.file_path,
                    status=StemStatus.READY if r.status == "ready" else StemStatus.FAILED,
                    error_message=r.error,
                )
            )

    if result.failed_count > 0:
        log.warning("⚠ %d stems failed; %d ready", result.failed_count, result.success_count)

    # Guard: stitch_and_master_audio needs ≥ 2 READY stems to produce a mix.
    # If the batch yielded fewer, fail fast here rather than letting DSP crash
    # on a RuntimeError after retrying 3 × 30 s for nothing.
    if result.success_count < 2:
        msg = (
            f"Insufficient READY stems: {result.success_count}/{len(batch.prompts)} "
            f"generated successfully — cannot stitch a mix."
        )
        _mark_mix_failed(mix_id, msg)
        raise RuntimeError(msg)

    log.info("✅ Stems ready: %d/%d", result.success_count, len(batch.prompts))
    # Pass only what downstream tasks need
    return {
        "mix_id": mix_id,
        "stem_count": result.success_count,
        "bpm": batch.strategy.bpm,
        "mastered_audio_path": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 — DSP: Stitch & Master
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.stitch_and_master_audio",
    bind=True,
    queue="dsp",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    soft_time_limit=7200,   # Full mix can take up to 2h of DSP on slow CPU
    time_limit=7500,
)
def stitch_and_master_audio(self, upstream: dict) -> dict:
    """
    DSP Worker core task.
    1. Loads all READY stems for the mix from disk.
    2. Uses librosa to detect beats and phase-align each stem.
    3. Crossfades only on the 'one' of the beat grid.
    4. Applies pedalboard mastering chain: EQ → Sidechain → Limiter (-14 LUFS).
    5. Exports WAV + MP3 to MIXES_DIR.
    """
    mix_id = upstream["mix_id"]
    log.info("🎛  DSP stitch+master: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.STITCHING)

    try:
        from services.dsp_worker.audio.stem_stitcher import stitch_and_master

        result: AudioStitchResult = stitch_and_master(mix_id=mix_id)
    except Exception as exc:
        log.error("DSP stitch error: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"DSP stitching failed: {exc}")
            raise

    with get_sync_db() as db:
        db.execute(
            update(Mix).where(Mix.id == mix_id).values(
                mastered_audio_path=result.mastered_audio_path,
                actual_duration_seconds=result.actual_duration_seconds,
                lufs_measured=result.lufs_integrated,
                true_peak_dbfs=result.true_peak_dbfs,
                status=MixStatus.MASTERING,
            )
        )

    log.info("✅ Mastered: path=%s LUFS=%.1f TP=%.1f",
             result.mastered_audio_path, result.lufs_integrated, result.true_peak_dbfs)
    return result.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 — DSP: QA Audio Check
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.run_qa_audio_check",
    bind=True,
    queue="dsp",
    max_retries=2,
    default_retry_delay=15,
    acks_late=True,
)
def run_qa_audio_check(self, stitch_result: dict) -> dict:
    """
    QA & Auto-Heal agent (audio phase).
    Verifies that the mastered file hits -14 LUFS ±0.5 LU and true peak ≤ -1 dBFS.
    On failure, triggers a re-master with adjusted limiter threshold.
    """
    result = AudioStitchResult(**stitch_result)
    mix_id = result.mix_id
    log.info("🔬 QA audio: mix_id=%s LUFS=%.2f TP=%.2f", mix_id, result.lufs_integrated, result.true_peak_dbfs)
    _update_mix_status(mix_id, MixStatus.QA_AUDIO)

    from services.dsp_worker.audio.mastering import run_audio_qa

    qa: AudioQAResult = run_audio_qa(
        mix_id=mix_id,
        audio_path=result.mastered_audio_path,
        target_lufs=-14.0,
        true_peak_ceiling=-1.0,
    )

    with get_sync_db() as db:
        db.execute(
            update(Mix).where(Mix.id == mix_id).values(
                lufs_measured=qa.lufs_measured,
                true_peak_dbfs=qa.true_peak_dbfs,
                qa_passed=qa.passed,
            )
        )

    if not qa.passed:
        issues_str = "; ".join(qa.issues)
        log.warning("⚠ QA audio FAIL (attempt %d): %s", self.request.retries + 1, issues_str)
        try:
            raise self.retry(
                exc=ValueError(f"Audio QA failed: {issues_str}"),
                countdown=15,
            )
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Audio QA failed permanently: {issues_str}")
            raise

    log.info("✅ QA audio passed: LUFS=%.2f TP=%.2f", qa.lufs_measured, qa.true_peak_dbfs)
    # Pass audio path forward for video rendering
    return {**stitch_result, "qa_audio_passed": True}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 — Visual Prompt Engineer
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.generate_visual_prompts",
    bind=True,
    queue="orchestration",
    max_retries=3,
    default_retry_delay=15,
    acks_late=True,
)
def generate_visual_prompts(self, upstream: dict) -> dict:
    """
    Visual Prompt Engineer agent.
    Generates English prompts for:
      - 1× 16:9 background image (Nano Banana 2)
      - 3× 16:9 Veo video loops  (cinematic ambience)
      - 1× 9:16 Short background  (Nano Banana 2)
      - 3× 9:16 Short thumbnails  (one per viral clip)
    All prompts are in English.
    """
    mix_id = upstream["mix_id"]
    log.info("🖼  Visual prompts: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.FETCHING_VISUALS)

    try:
        from services.orchestrator.crew.agents import run_visual_prompt_engineer

        with get_sync_db() as db:
            mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

        batch: VisualPromptBatch = run_visual_prompt_engineer(
            mix_id=mix_id,
            subgenre=mix.subgenre,
            bpm=mix.bpm,
            style_hint=mix.style_hint,
        )
    except Exception as exc:
        log.warning("Visual prompt error: %s", exc)
        if _is_quota_exhausted(exc):
            _mark_mix_failed(mix_id, f"OpenAI quota exhausted — add credits: {exc}")
            raise
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 10)
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Visual prompt generation failed: {exc}")
            raise

    with get_sync_db() as db:
        for p in batch.prompts:
            visual = Visual(
                mix_id=mix_id,
                visual_type=p.visual_type,
                aspect_ratio=p.aspect_ratio,
                gemini_prompt=p.prompt_en,
                status=VisualStatus.PENDING,
            )
            db.add(visual)

    log.info("✅ Visual prompts: %d assets", len(batch.prompts))
    return {**upstream, "visual_prompt_count": len(batch.prompts)}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 — Gemini Visual Fetcher
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.fetch_visuals_from_gemini",
    bind=True,
    queue="orchestration",
    max_retries=4,
    acks_late=True,
    soft_time_limit=1800,
    time_limit=1900,
)
def fetch_visuals_from_gemini(self, upstream: dict) -> dict:
    """
    Calls Gemini Nano Banana 2 (images) and Veo (video loops).
    Downloads artefacts to VISUALS_DIR/{mix_id}/.
    Tenacity handles per-request rate-limit backoff inside the agent.
    """
    mix_id = upstream["mix_id"]
    log.info("🌐 Fetching visuals from Gemini: mix_id=%s", mix_id)

    try:
        from services.orchestrator.crew.agents import fetch_visuals_batch

        fetch_visuals_batch(mix_id=mix_id)
    except Exception as exc:
        log.warning("Visual fetch error: %s", exc)
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 20)
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Visual fetching failed: {exc}")
            raise

    log.info("✅ Visuals fetched: mix_id=%s", mix_id)
    return upstream


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 — Video: Render Full Mix
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.render_full_video",
    bind=True,
    queue="video",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    soft_time_limit=14400,   # 4-hour soft limit for 4K render
    time_limit=14700,
)
def render_full_video(self, upstream: dict) -> dict:
    """
    Video Worker: FFmpeg hardware-accelerated renderer.
    Composites the mastered audio with:
      - Seamlessly looped Veo video backgrounds
      - Dynamic audio spectrum analyser overlay (complex filtergraph)
      - Chapter title overlays timed to the musical arc
    Output: EXPORTS_DIR/{mix_id}/full_mix.mp4  (3840×2160 or 1920×1080)
    """
    mix_id = upstream["mix_id"]
    log.info("🎬 Render full video: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.RENDERING)

    try:
        from services.video_worker.video.renderer import render_full_mix_video

        result: VideoRenderResult = render_full_mix_video(mix_id=mix_id)
    except Exception as exc:
        log.error("Video render error: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Video render failed: {exc}")
            raise

    with get_sync_db() as db:
        db.execute(
            update(Mix).where(Mix.id == mix_id).values(full_video_path=result.full_video_path)
        )

    log.info("✅ Video rendered: %s [%s %s]", result.full_video_path, result.resolution, result.codec)
    return {**upstream, **result.model_dump()}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 — Video: Slice Viral Shorts
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.slice_viral_shorts",
    bind=True,
    queue="video",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    soft_time_limit=3600,
    time_limit=3700,
)
def slice_viral_shorts(self, upstream: dict) -> dict:
    """
    Content Slicer (Viral Agent).
    1. Analyses the mastered audio waveform for RMS energy using librosa.
    2. Finds the 3 non-overlapping 60-second windows with highest RMS (loudest drops).
    3. Renders each as a 9:16 1080×1920 vertical video with:
       - Matching background from Shorts visuals
       - Animated spectrum overlay
    Output: EXPORTS_DIR/{mix_id}/short_{rank}.mp4
    """
    mix_id = upstream["mix_id"]
    log.info("✂  Slice viral shorts: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.SLICING)

    try:
        from services.video_worker.video.slicer import slice_viral_shorts_from_mix

        result: ViralSliceResult = slice_viral_shorts_from_mix(mix_id=mix_id)
    except Exception as exc:
        log.error("Slice error: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Viral short slicing failed: {exc}")
            raise

    with get_sync_db() as db:
        for s in result.shorts:
            short = ViralShort(
                id=s.short_id,
                mix_id=mix_id,
                rank=s.rank,
                start_seconds=s.start_seconds,
                rms_db=s.rms_db,
                video_path=s.video_path,
            )
            db.add(short)

    log.info("✅ Viral shorts sliced: %d clips", len(result.shorts))
    return {**upstream, "viral_shorts": [s.model_dump() for s in result.shorts]}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 10 — Video: QA Check
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.run_qa_video_check",
    bind=True,
    queue="video",
    max_retries=2,
    default_retry_delay=20,
    acks_late=True,
)
def run_qa_video_check(self, upstream: dict) -> dict:
    """
    QA & Auto-Heal agent (video phase).
    Probes the rendered MP4 with ffprobe to detect:
      - Frame drops (pts_discontinuity > 0.5%)
      - Audio/video sync drift (> 40 ms)
      - Corrupt keyframes
    On failure, re-queues render_full_video with corrected FFmpeg flags.
    """
    mix_id = upstream["mix_id"]
    log.info("🔬 QA video: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.QA_VIDEO)

    from services.video_worker.video.renderer import run_video_qa

    qa: VideoQAResult = run_video_qa(mix_id=mix_id)

    if not qa.passed:
        issues_str = "; ".join(qa.issues)
        log.warning("⚠ QA video FAIL: %s", issues_str)
        try:
            raise self.retry(
                exc=ValueError(f"Video QA failed: {issues_str}"),
                countdown=20,
            )
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Video QA failed permanently: {issues_str}")
            raise

    log.info("✅ QA video passed: frame_drop_rate=%.4f%%", qa.frame_drop_rate * 100)
    return {**upstream, "qa_video_passed": True}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 11 — Polish SEO & Growth Hacker
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.generate_polish_seo",
    bind=True,
    queue="orchestration",
    max_retries=3,
    default_retry_delay=20,
    acks_late=True,
)
def generate_polish_seo(self, upstream: dict) -> dict:
    """
    Polish SEO & Growth Hacker agent.

    ALL output is in hyper-optimised, viral Polish (język polski):
      - title_pl:       YouTube / TikTok title (≤ 100 chars, clickbait hooks)
      - description_pl: Full SEO description with keywords, timestamps (≤ 5000 chars)
      - tags_pl:        Hyper-targeted Polish hashtags (≤ 500 chars total)
      - chapters_pl:    Precise chapter markers synced to musical arc
      - shorts_titles_pl: 3 short titles — one per viral clip

    The backend prompt to CrewAI/GPT-4o is in English.
    The model is instructed to output exclusively in Polish.
    """
    mix_id = upstream["mix_id"]
    log.info("🇵🇱 Polish SEO: mix_id=%s", mix_id)

    try:
        from services.orchestrator.crew.agents import run_polish_seo_agent

        with get_sync_db() as db:
            mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

        seo: PolishSEOMetadata = run_polish_seo_agent(
            mix_id=mix_id,
            bpm=mix.bpm,
            subgenre=mix.subgenre,
            style_hint=mix.style_hint,
            actual_duration_seconds=mix.actual_duration_seconds,
        )
    except Exception as exc:
        log.warning("SEO generation error: %s", exc)
        if _is_quota_exhausted(exc):
            _mark_mix_failed(mix_id, f"OpenAI quota exhausted — add credits: {exc}")
            raise
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 10)
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"Polish SEO generation failed: {exc}")
            raise

    log.info("✅ Polish SEO ready: title='%s'", seo.title_pl[:60])
    return {**upstream, "seo": seo.model_dump()}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 12 — Upload to YouTube
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.upload_to_youtube",
    bind=True,
    queue="upload",
    max_retries=5,
    acks_late=True,
)
def upload_to_youtube(self, upstream: dict) -> dict:
    """
    Uploads:
      1. The full 16:9 mix video with Polish metadata + chapters.
      2. Up to 3 YouTube Shorts (9:16, 60s, Polish titles).
    Uses exponential backoff (tenacity) for YouTube quota handling.
    """
    mix_id = upstream["mix_id"]
    log.info("📤 YouTube upload: mix_id=%s", mix_id)
    _update_mix_status(mix_id, MixStatus.UPLOADING)

    try:
        from services.orchestrator.crew.agents import upload_to_youtube_agent

        results: list[UploadResult] = upload_to_youtube_agent(
            mix_id=mix_id,
            seo=upstream.get("seo", {}),
            viral_shorts=upstream.get("viral_shorts", []),
        )
    except Exception as exc:
        log.warning("YouTube upload error (attempt %d): %s", self.request.retries + 1, exc)
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 30)
        except MaxRetriesExceededError:
            _mark_mix_failed(mix_id, f"YouTube upload failed after retries: {exc}")
            raise

    with get_sync_db() as db:
        for r in results:
            upload_record = PlatformUpload(
                mix_id=mix_id,
                platform=Platform.YOUTUBE,
                content_type=r.content_type,
                upload_status=UploadStatus.UPLOADED if r.status == "uploaded" else UploadStatus.FAILED,
                platform_video_id=r.platform_video_id,
                platform_video_url=r.platform_video_url,
                title_pl=upstream.get("seo", {}).get("title_pl"),
                description_pl=upstream.get("seo", {}).get("description_pl"),
                tags_pl=upstream.get("seo", {}).get("tags_pl"),
                chapters_pl=upstream.get("seo", {}).get("chapters_pl"),
                error_message=r.error,
            )
            db.add(upload_record)

    log.info("✅ YouTube uploads: %d completed", len(results))
    return {**upstream, "youtube_uploads": [r.model_dump() for r in results]}


# ─────────────────────────────────────────────────────────────────────────────
# TASK 13 — Upload to TikTok
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="shared.tasks.definitions.upload_to_tiktok",
    bind=True,
    queue="upload",
    max_retries=5,
    acks_late=True,
)
def upload_to_tiktok(self, upstream: dict) -> dict:
    """
    Uploads the 3 viral 9:16 shorts to TikTok with Polish titles and hashtags.
    Full-mix content is YouTube-only (TikTok 10-min limit).
    """
    mix_id = upstream["mix_id"]
    viral_shorts = upstream.get("viral_shorts", [])
    log.info("📤 TikTok upload: mix_id=%s shorts=%d", mix_id, len(viral_shorts))

    try:
        from services.orchestrator.crew.agents import upload_to_tiktok_agent

        results: list[UploadResult] = upload_to_tiktok_agent(
            mix_id=mix_id,
            seo=upstream.get("seo", {}),
            viral_shorts=viral_shorts,
        )
    except Exception as exc:
        log.warning("TikTok upload error (attempt %d): %s", self.request.retries + 1, exc)
        try:
            raise self.retry(exc=exc, countdown=int(2 ** self.request.retries) * 30)
        except MaxRetriesExceededError:
            # TikTok failure is non-fatal — log but do not abort entire pipeline
            log.error("TikTok upload PERMANENTLY failed for mix %s: %s", mix_id, exc)
            _update_mix_status(mix_id, MixStatus.COMPLETE)
            return {**upstream, "tiktok_uploads": [], "warning": str(exc)}

    with get_sync_db() as db:
        for r in results:
            upload_record = PlatformUpload(
                mix_id=mix_id,
                platform=Platform.TIKTOK,
                content_type=r.content_type,
                upload_status=UploadStatus.UPLOADED if r.status == "uploaded" else UploadStatus.FAILED,
                platform_video_id=r.platform_video_id,
                platform_video_url=r.platform_video_url,
                title_pl=upstream.get("seo", {}).get("shorts_titles_pl", [""])[0],
                tags_pl=upstream.get("seo", {}).get("tags_pl"),
                error_message=r.error,
            )
            db.add(upload_record)

        # Mark the entire pipeline complete
        db.execute(update(Mix).where(Mix.id == mix_id).values(status=MixStatus.COMPLETE))

    log.info("🎉 Pipeline COMPLETE: mix_id=%s TikTok uploads=%d", mix_id, len(results))
    return {**upstream, "tiktok_uploads": [r.model_dump() for r in results], "pipeline_status": "complete"}
