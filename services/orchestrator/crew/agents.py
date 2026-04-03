"""
services/orchestrator/crew/agents.py
─────────────────────────────────────────────────────────────────────────────
Public entry-point functions for all Project Nebula agents.
These are the functions imported and called by shared/tasks/definitions.py.

Each function:
  1. Builds/runs the appropriate CrewAI crew OR delegates to a MediaGenerator
  2. Parses the structured output into a typed Pydantic schema
  3. Returns the schema object — no raw dicts leak out

Media generation is provider-agnostic — controlled by env vars:
  AUDIO_PROVIDER=replicate (default) | gemini
  IMAGE_PROVIDER=dalle3    (default) | gemini
  VIDEO_PROVIDER=ffmpeg    (default) | gemini

All media is written to the shared volume mounts defined in Settings.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import httpx  # TikTok upload
from sqlalchemy import select, update

from shared.config import get_settings
from shared.db.models import (
    BpmSubgenreRegistry,
    ContentType,
    Mix,
    Platform,
    Stem,
    StemStatus,
    Visual,
    VisualStatus,
)
from shared.db.session import get_sync_db
from shared.schemas.events import (
    AudioPromptBatch,
    CSOStrategy,
    PolishSEOMetadata,
    StemBatchResult,
    StemFetchResult,
    StemPrompt,
    UploadResult,
    VisualPrompt,
    VisualPromptBatch,
)
from shared.media.factory import get_audio_generator, get_image_generator, get_video_generator
from shared.utils.retry import retry_openai_api, retry_youtube_api, retry_tiktok_api

log = logging.getLogger("nebula.agents")
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_crew_json(raw_output: str, schema_hint: str = "") -> dict:
    """
    Extract a JSON object from a CrewAI agent's raw string output.

    Handles three common agent output patterns:
      1. Bare JSON                     → parse directly
      2. ```json\\n{...}\\n```          → strip fences, parse
      3. Prose text + ```json\\n{...}  → extract the code block, parse
    """
    import re

    text = raw_output.strip()

    # Pattern 1: try to parse as-is first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pattern 2 & 3: extract content from any ```[json] ... ``` block
    code_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if code_match:
        candidate = code_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Pattern 4: find the first { or [ and try from there
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                break

    log.error("JSON parse error in crew output (%s)\nRaw: %.500s", schema_hint, raw_output)
    raise ValueError(f"Agent returned invalid JSON for {schema_hint}")


def _stems_dir(mix_id: str) -> Path:
    p = Path(settings.stems_dir) / mix_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _visuals_dir(mix_id: str) -> Path:
    p = Path(settings.visuals_dir) / mix_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _exports_dir(mix_id: str) -> Path:
    p = Path(settings.exports_dir) / mix_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSO Agent
# ─────────────────────────────────────────────────────────────────────────────

@retry_openai_api
def run_cso_agent(
    mix_id: str,
    genre: str,
    requested_duration_minutes: int,
) -> CSOStrategy:
    """
    Runs the Chief Strategy Officer CrewAI crew.
    Genre is the only user input; all other parameters (BPM, subgenre, key,
    arc, stem count) are decided autonomously by the CSO agent.
    Queries the dedup registry so the agent never repeats a combination.
    """
    from services.orchestrator.crew.crew import build_strategy_crew
    from shared.genres import GENRES

    # Pass genre profile to the CSO so it knows the BPM + duration bounds
    genre_profile = GENRES.get(genre, {})

    # Load the full registry to pass as context to the CSO
    with get_sync_db() as db:
        rows = db.execute(
            select(
                BpmSubgenreRegistry.bpm,
                BpmSubgenreRegistry.subgenre,
                BpmSubgenreRegistry.key_signature,
            )
        ).fetchall()

    used_combinations = [
        {"bpm": r.bpm, "subgenre": r.subgenre, "key_signature": r.key_signature}
        for r in rows
    ]
    used_json = json.dumps(used_combinations)

    crew = build_strategy_crew(
        mix_id=mix_id,
        genre=genre,
        genre_bpm_range=genre_profile.get("bpm_range", (100, 180)),
        requested_duration_minutes=requested_duration_minutes,
        used_combinations_json=used_json,
    )

    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)
    data = _parse_crew_json(raw, "CSOStrategy")
    data["mix_id"] = mix_id  # Ensure mix_id is present

    log.info("CSO output: bpm=%.1f subgenre=%s", data.get("bpm"), data.get("subgenre"))
    return CSOStrategy(**data)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audio Prompt Engineer
# ─────────────────────────────────────────────────────────────────────────────

@retry_openai_api
def run_audio_prompt_engineer(strategy: CSOStrategy) -> AudioPromptBatch:
    """
    Runs the Audio Prompt Engineer crew to generate stem_count English
    Lyria 3 prompts forming a coherent musical arc.
    """
    from services.orchestrator.crew.crew import build_audio_prompt_crew

    crew = build_audio_prompt_crew(
        mix_id=strategy.mix_id,
        bpm=strategy.bpm,
        subgenre=strategy.subgenre,
        key_signature=strategy.key_signature,
        style_description=strategy.style_description,
        transition_arc=strategy.transition_arc,
        stem_count=strategy.stem_count,
    )

    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)
    data = _parse_crew_json(raw, "AudioPromptBatch")
    data["mix_id"] = strategy.mix_id
    data["strategy"] = strategy.model_dump()

    # Validate each prompt entry
    prompts = [StemPrompt(**p) for p in data.get("prompts", [])]
    log.info("Audio prompts generated: %d stems for mix %s", len(prompts), strategy.mix_id)
    return AudioPromptBatch(mix_id=strategy.mix_id, strategy=strategy, prompts=prompts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stem Fetcher — provider-agnostic via MediaGenerator factory
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stems_batch(batch: AudioPromptBatch) -> StemBatchResult:
    """
    Generates every stem in the batch via the configured AudioGenerator.
    Provider is selected at runtime from AUDIO_PROVIDER env var:
      replicate → ReplicateMusicGenProvider (~$0.008/stem, production-ready)
      gemini    → GeminiLyriaProvider (experimental, requires waitlist access)

    Files are written as WAV to STEMS_DIR/{mix_id}/{position:04d}.wav
    """
    mix_id    = batch.strategy.mix_id
    out_dir   = _stems_dir(mix_id)
    generator = get_audio_generator()
    results: list[StemFetchResult] = []

    log.info("Generating %d stems via %s", len(batch.prompts), generator.provider_name)

    # Load existing stem IDs from DB (needed for result records)
    with get_sync_db() as db:
        stem_rows = db.execute(
            select(Stem.id, Stem.position).where(Stem.mix_id == mix_id)
        ).fetchall()
    stem_id_map = {r.position: r.id for r in stem_rows}

    for prompt in batch.prompts:
        stem_id   = stem_id_map.get(prompt.position, str(uuid.uuid4()))
        file_path = str(out_dir / f"{prompt.position:04d}.wav")

        try:
            generator.generate_stem(
                prompt=prompt.prompt_en,
                bpm=batch.strategy.bpm,
                duration_s=settings.lyria_stem_duration_seconds,
                output_path=file_path,
            )
            with get_sync_db() as db:
                db.execute(
                    update(Stem)
                    .where(Stem.mix_id == mix_id, Stem.position == prompt.position)
                    .values(status=StemStatus.READY, file_path=file_path)
                )
            results.append(StemFetchResult(
                mix_id=mix_id, stem_id=stem_id,
                position=prompt.position, file_path=file_path, status="ready",
            ))
            log.debug("Stem %04d ready: %s", prompt.position, file_path)

        except Exception as exc:
            log.error("Stem %04d FAILED: %s", prompt.position, exc)
            with get_sync_db() as db:
                db.execute(
                    update(Stem)
                    .where(Stem.mix_id == mix_id, Stem.position == prompt.position)
                    .values(status=StemStatus.FAILED, error_message=str(exc)[:1000])
                )
            results.append(StemFetchResult(
                mix_id=mix_id, stem_id=stem_id,
                position=prompt.position, file_path="", status="failed", error=str(exc),
            ))

    return StemBatchResult(mix_id=mix_id, results=results)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visual Prompt Engineer
# ─────────────────────────────────────────────────────────────────────────────

@retry_openai_api
def run_visual_prompt_engineer(
    mix_id: str,
    subgenre: str | None,
    bpm: float | None,
    style_hint: str | None,
) -> VisualPromptBatch:
    """
    Runs the Visual Prompt Engineer crew to generate image/video prompts.
    """
    from services.orchestrator.crew.crew import build_visual_prompt_crew

    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

    crew = build_visual_prompt_crew(
        mix_id=mix_id,
        subgenre=mix.subgenre or subgenre or "Drum and Bass",
        bpm=mix.bpm or bpm or 174.0,
        style_description=style_hint or f"{mix.subgenre} drum and bass mix at {mix.bpm} BPM",
    )

    result = crew.kickoff()
    raw  = result.raw if hasattr(result, "raw") else str(result)
    data = _parse_crew_json(raw, "VisualPromptBatch")
    data["mix_id"] = mix_id

    prompts = [VisualPrompt(**p) for p in data.get("prompts", [])]
    log.info("Visual prompts generated: %d assets for mix %s", len(prompts), mix_id)
    return VisualPromptBatch(mix_id=mix_id, prompts=prompts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visual Fetcher — provider-agnostic via MediaGenerator factory
# ─────────────────────────────────────────────────────────────────────────────

def fetch_visuals_batch(mix_id: str) -> None:
    """
    Generates all pending Visual rows via the configured Image/VideoGenerator.

    Provider selection (env vars):
      IMAGE_PROVIDER=dalle3   → DallE3Provider      (~$0.080/image, HD vivid)
      IMAGE_PROVIDER=gemini   → GeminiImagenProvider (experimental)
      VIDEO_PROVIDER=ffmpeg   → FFmpegKenBurnsProvider (free, animated Ken Burns)
      VIDEO_PROVIDER=gemini   → GeminiVeoProvider    (experimental)

    For video_loop visuals, the pipeline first generates a static image and
    then animates it with the VideoGenerator. This guarantees that even the
    FFmpeg Ken Burns provider produces visually rich, content-coherent loops.
    """
    with get_sync_db() as db:
        pending = db.execute(
            select(Visual).where(
                Visual.mix_id == mix_id,
                Visual.status == VisualStatus.PENDING,
            )
        ).scalars().all()

    out_dir       = _visuals_dir(mix_id)
    img_generator = get_image_generator()
    vid_generator = get_video_generator()

    log.info(
        "Generating %d visuals — img:%s vid:%s",
        len(pending), img_generator.provider_name, vid_generator.provider_name,
    )

    for visual in pending:
        visual_id = str(visual.id)
        try:
            if visual.visual_type.value == "video_loop":
                # Step 1: generate a source image
                src_image_path = str(out_dir / f"{visual_id}_src.png")
                img_generator.generate_image(
                    prompt=visual.gemini_prompt,
                    aspect_ratio=visual.aspect_ratio,
                    output_path=src_image_path,
                )
                # Step 2: animate into a video loop
                output_path = str(out_dir / f"{visual_id}.mp4")
                vid_generator.generate_video_loop(
                    prompt=visual.gemini_prompt,
                    aspect_ratio=visual.aspect_ratio,
                    output_path=output_path,
                    source_image_path=src_image_path,
                )
                file_path = output_path
            else:
                # Static image (background_still, album_art, short_cover, etc.)
                ext = "png"
                output_path = str(out_dir / f"{visual_id}.{ext}")
                img_generator.generate_image(
                    prompt=visual.gemini_prompt,
                    aspect_ratio=visual.aspect_ratio,
                    output_path=output_path,
                )
                file_path = output_path

            with get_sync_db() as db:
                db.execute(
                    update(Visual)
                    .where(Visual.id == visual.id)
                    .values(file_path=file_path, status=VisualStatus.READY)
                )
            log.debug("Visual %s ready: %s", visual_id, file_path)

        except Exception as exc:
            log.error("Visual %s FAILED: %s", visual_id, exc)
            with get_sync_db() as db:
                db.execute(
                    update(Visual)
                    .where(Visual.id == visual.id)
                    .values(status=VisualStatus.FAILED, error_message=str(exc)[:1000])
                )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Polish SEO Agent
# ─────────────────────────────────────────────────────────────────────────────

@retry_openai_api
def run_polish_seo_agent(
    mix_id: str,
    bpm: float | None,
    subgenre: str | None,
    style_hint: str | None,
    actual_duration_seconds: float | None,
) -> PolishSEOMetadata:
    """
    Runs the Polish SEO & Growth Hacker CrewAI crew.
    ALL output is in Polish (język polski).
    """
    from services.orchestrator.crew.crew import build_seo_crew

    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

    crew = build_seo_crew(
        mix_id=mix_id,
        bpm=mix.bpm or bpm or 174.0,
        subgenre=mix.subgenre or subgenre or "Drum and Bass",
        style_description=mix.style_hint or style_hint or "",
        transition_arc="",  # Could be stored in Mix if needed
        actual_duration_seconds=mix.actual_duration_seconds or actual_duration_seconds or 2700.0,
        style_hint=mix.style_hint,
    )

    result = crew.kickoff()
    raw  = result.raw if hasattr(result, "raw") else str(result)
    data = _parse_crew_json(raw, "PolishSEOMetadata")
    data["mix_id"] = mix_id

    log.info("Polish SEO ready: title='%s'", data.get("title_pl", "")[:60])
    return PolishSEOMetadata(**data)


# ─────────────────────────────────────────────────────────────────────────────
# 7. YouTube Upload Agent
# ─────────────────────────────────────────────────────────────────────────────

@retry_youtube_api
def upload_to_youtube_agent(
    mix_id: str,
    seo: dict,
    viral_shorts: list[dict],
) -> list[UploadResult]:
    """
    Uploads the full mix video and YouTube Shorts to YouTube Data API v3.
    Uses OAuth2 service account credentials from the mounted secrets file.

    Returns a list of UploadResult objects — one per upload attempt.
    """
    from googleapiclient.discovery import build as yt_build
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.credentials import Credentials

    results: list[UploadResult] = []
    exports = _exports_dir(mix_id)

    # ── Build YouTube API client ──────────────────────────────────────────
    # Uses the pre-authorized token file (generated once via scripts/youtube_auth.py).
    # This file contains a refresh_token and is distinct from client_secrets.json.
    try:
        creds = Credentials.from_authorized_user_file(
            settings.youtube_token_file,   # /secrets/youtube_token.json
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )
        # Auto-refresh the access token if expired (requires refresh_token in file)
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            # Persist refreshed token back to disk
            import json as _json
            with open(settings.youtube_token_file, "w") as _tf:
                _json.dump({
                    "token":         creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri":     creds.token_uri,
                    "client_id":     creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes":        list(creds.scopes or []),
                }, _tf)
        youtube = yt_build("youtube", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        log.error("YouTube auth failed: %s", exc)
        raise

    # ── Upload full mix (16:9) ────────────────────────────────────────────
    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

    if mix.full_video_path and os.path.exists(mix.full_video_path):
        chapters_body = "\n".join(
            f"{c['time_str']} {c['title_pl']}"
            for c in (seo.get("chapters_pl") or [])
        )
        full_description = (
            f"{seo.get('description_pl', '')}\n\n"
            f"📍 ROZDZIAŁY:\n{chapters_body}"
        )

        body = {
            "snippet": {
                "title":       seo.get("title_pl", "")[:100],
                "description": full_description[:5000],
                "tags":        (seo.get("tags_pl") or [])[:500],
                "categoryId":  "10",  # Music
                "defaultLanguage": "pl",
            },
            "status": {
                "privacyStatus":      "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        try:
            media = MediaFileUpload(
                mix.full_video_path,
                mimetype="video/mp4",
                resumable=True,
                chunksize=10 * 1024 * 1024,  # 10 MB chunks
            )
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )
            response = _execute_resumable_upload(request)
            video_id = response.get("id", "")
            results.append(UploadResult(
                mix_id=mix_id,
                upload_id=str(uuid.uuid4()),
                platform="youtube",
                content_type=ContentType.FULL_MIX.value,
                platform_video_id=video_id,
                platform_video_url=f"https://www.youtube.com/watch?v={video_id}",
                status="uploaded",
            ))
            log.info("YouTube full mix uploaded: https://youtu.be/%s", video_id)
        except Exception as exc:
            log.error("YouTube full mix upload failed: %s", exc)
            results.append(UploadResult(
                mix_id=mix_id, upload_id=str(uuid.uuid4()),
                platform="youtube", content_type=ContentType.FULL_MIX.value,
                status="failed", error=str(exc),
            ))

    # ── Upload YouTube Shorts (9:16) ──────────────────────────────────────
    shorts_titles = seo.get("shorts_titles_pl") or []
    for i, short in enumerate(viral_shorts):
        video_path = short.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            continue

        title = (shorts_titles[i] if i < len(shorts_titles) else seo.get("title_pl", ""))[:100]

        short_body = {
            "snippet": {
                "title":       title,
                "description": seo.get("description_pl", "")[:5000],
                "tags":        (seo.get("tags_pl") or [])[:500],
                "categoryId":  "10",
                "defaultLanguage": "pl",
            },
            "status": {
                "privacyStatus":      "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        try:
            media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
            request = youtube.videos().insert(
                part="snippet,status",
                body=short_body,
                media_body=media,
            )
            resp     = _execute_resumable_upload(request)
            video_id = resp.get("id", "")
            results.append(UploadResult(
                mix_id=mix_id, upload_id=str(uuid.uuid4()),
                platform="youtube", content_type=ContentType.SHORT.value,
                platform_video_id=video_id,
                platform_video_url=f"https://www.youtube.com/shorts/{video_id}",
                status="uploaded",
            ))
            log.info("YouTube Short #%d uploaded: https://youtube.com/shorts/%s", i + 1, video_id)
        except Exception as exc:
            log.error("YouTube Short #%d failed: %s", i + 1, exc)
            results.append(UploadResult(
                mix_id=mix_id, upload_id=str(uuid.uuid4()),
                platform="youtube", content_type=ContentType.SHORT.value,
                status="failed", error=str(exc),
            ))

    return results


def _execute_resumable_upload(request) -> dict:
    """Drive a resumable YouTube upload to completion."""
    response = None
    while response is None:
        _, response = request.next_chunk()
    return response


# ─────────────────────────────────────────────────────────────────────────────
# 8. TikTok Upload Agent
# ─────────────────────────────────────────────────────────────────────────────

@retry_tiktok_api
def upload_to_tiktok_agent(
    mix_id: str,
    seo: dict,
    viral_shorts: list[dict],
) -> list[UploadResult]:
    """
    Uploads viral shorts to TikTok via the TikTok Content Posting API v2.
    Only 9:16 short clips are uploaded — TikTok has a 10-minute limit.
    """
    results: list[UploadResult] = []
    shorts_titles = seo.get("shorts_titles_pl") or []
    tags_pl       = " ".join(seo.get("tags_pl") or [])[:150]

    for i, short in enumerate(viral_shorts):
        video_path = short.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            continue

        title = (shorts_titles[i] if i < len(shorts_titles) else "")[:80]
        caption = f"{title}\n{tags_pl}"

        try:
            video_id = _tiktok_upload_video(
                video_path=video_path,
                caption=caption,
                api_key=settings.tiktok_api_key,
            )
            results.append(UploadResult(
                mix_id=mix_id, upload_id=str(uuid.uuid4()),
                platform="tiktok", content_type=ContentType.SHORT.value,
                platform_video_id=video_id,
                platform_video_url=f"https://www.tiktok.com/@nebula/video/{video_id}",
                status="uploaded",
            ))
            log.info("TikTok Short #%d uploaded: video_id=%s", i + 1, video_id)
        except Exception as exc:
            log.error("TikTok Short #%d failed: %s", i + 1, exc)
            results.append(UploadResult(
                mix_id=mix_id, upload_id=str(uuid.uuid4()),
                platform="tiktok", content_type=ContentType.SHORT.value,
                status="failed", error=str(exc),
            ))

    return results


@retry_tiktok_api
def _tiktok_upload_video(video_path: str, caption: str, api_key: str) -> str:
    """
    TikTok Content Posting API v2 — direct post upload flow.
    Returns the TikTok publish_id on success.
    """
    file_size = os.path.getsize(video_path)

    # Step 1: Initialise upload
    init_resp = httpx.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={
            "post_info": {
                "title":          caption[:150],
                "privacy_level":  "PUBLIC_TO_EVERYONE",
                "disable_comment": False,
                "disable_duet":    False,
                "disable_stitch":  False,
            },
            "source_info": {
                "source":     "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": file_size,
                "total_chunk_count": 1,
            },
        },
        timeout=30.0,
    )
    init_resp.raise_for_status()
    init_data   = init_resp.json()["data"]
    upload_url  = init_data["upload_url"]
    publish_id  = init_data["publish_id"]

    # Step 2: Upload the video file
    with open(video_path, "rb") as f:
        video_data = f.read()

    upload_resp = httpx.put(
        upload_url,
        content=video_data,
        headers={
            "Content-Type":  "video/mp4",
            "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        },
        timeout=300.0,
    )
    upload_resp.raise_for_status()

    # Step 3: Poll publish status
    for _ in range(30):
        status_resp = httpx.post(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json; charset=UTF-8"},
            json={"publish_id": publish_id},
            timeout=30.0,
        )
        status_resp.raise_for_status()
        status = status_resp.json()["data"]["status"]
        if status == "PUBLISH_COMPLETE":
            return publish_id
        if status in ("FAILED", "PUBLISH_FAILED"):
            raise RuntimeError(f"TikTok publish failed: {status_resp.json()}")
        time.sleep(10)

    raise TimeoutError(f"TikTok publish timed out for publish_id={publish_id}")
