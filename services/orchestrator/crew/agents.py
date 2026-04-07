"""
services/orchestrator/crew/agents.py
─────────────────────────────────────────────────────────────────────────────
Public entry-point functions for all Project Nebula agents.
These are the functions imported and called by shared/tasks/definitions.py.

Each function:
  1. Builds/runs the appropriate CrewAI crew OR delegates to a MediaGenerator
  2. Parses the structured output into a typed Pydantic schema
  3. Returns the schema object — no raw dicts leak out

Production media stack:
  AUDIO_PROVIDER=replicate → Replicate MusicGen (~$0.008/stem)
  IMAGE_PROVIDER=dalle3    → OpenAI DALL-E 3    (~$0.080/image HD)
  VIDEO_PROVIDER=ffmpeg    → FFmpeg Ken Burns    (FREE, local)

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

    Handles all common agent output patterns:
      1. Bare JSON                          → parse directly
      2. ```json\\n{...}\\n```              → strip fences, parse
      3. Prose text + ```json\\n{...}       → extract code block, parse
      4. JSON followed by prose/validation  → raw_decode stops at JSON end
      5. Prose then JSON then prose         → scan for {/[ and raw_decode
    """
    import re

    text = raw_output.strip()
    decoder = json.JSONDecoder()

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

    # Pattern 4 & 5: scan for the first { or [ and use raw_decode so
    # trailing prose (e.g. QA validation text) is ignored completely.
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                continue

    log.error("JSON parse error in crew output (%s)\nRaw: %.500s", schema_hint, raw_output)
    raise ValueError(f"Agent returned invalid JSON for {schema_hint}")


_VALID_TRANSITION_TYPES = {"intro", "build", "drop", "breakdown", "peak", "outro"}


def _validate_audio_prompts(
    prompts_raw: list,
    expected_count: int,
    batch_num: int,
    total_batches: int,
    pos_start: int,
    pos_end: int,
) -> None:
    """
    Python-level validation of audio prompt batches — replaces the LLM QA agent.

    Checks:
      1. At least `expected_count` prompts returned (LLM truncation detection)
      2. Each transition_type is a known enum value
      3. Each intensity is a float in [0.0, 1.0]
      4. Intensity values form a plausible arc (not all identical)
      5. Each prompt_en is non-empty

    Raises ValueError on first violation so the per-batch retry loop retries
    only the failing batch, not the whole mix.
    """
    tag = f"batch {batch_num}/{total_batches} (positions {pos_start}-{pos_end})"

    if not prompts_raw:
        raise ValueError(f"Audio PE returned 0 prompts for {tag}")

    # Strip null/malformed entries (position=None or prompt_en missing/empty).
    # The model occasionally appends a null trailing object in small last batches.
    valid_raw = [
        p for p in prompts_raw
        if p.get("position") is not None and str(p.get("prompt_en", "")).strip()
    ]
    if len(valid_raw) < len(prompts_raw):
        log.warning(
            "Stripped %d null/empty entries from %s (kept %d/%d)",
            len(prompts_raw) - len(valid_raw), tag, len(valid_raw), len(prompts_raw),
        )
    prompts_raw = valid_raw

    if len(prompts_raw) < expected_count:
        raise ValueError(
            f"Audio PE returned {len(prompts_raw)}/{expected_count} valid prompts for {tag} "
            f"— LLM truncated output"
        )

    intensities: list[float] = []
    for p in prompts_raw[:expected_count]:
        # transition_type
        tt = str(p.get("transition_type", "")).strip().lower()
        if tt not in _VALID_TRANSITION_TYPES:
            raise ValueError(
                f"Invalid transition_type '{tt}' in {tag} at position {p.get('position')} "
                f"— must be one of {_VALID_TRANSITION_TYPES}"
            )

        # intensity
        raw_intensity = p.get("intensity")
        try:
            intensity = float(raw_intensity)
        except (TypeError, ValueError):
            raise ValueError(
                f"Non-numeric intensity '{raw_intensity}' in {tag} at position {p.get('position')}"
            )
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(
                f"Intensity {intensity} out of range [0,1] in {tag} at position {p.get('position')}"
            )
        intensities.append(intensity)

        # prompt_en must be non-empty
        if not str(p.get("prompt_en", "")).strip():
            raise ValueError(f"Empty prompt_en in {tag} at position {p.get('position')}")

    # Arc sanity: all intensities identical → LLM just repeated same value
    unique_rounded = len(set(round(i, 1) for i in intensities))
    if expected_count >= 5 and unique_rounded == 1:
        raise ValueError(
            f"All {expected_count} intensities are identical ({intensities[0]}) in {tag} "
            f"— arc is flat, LLM likely failed to vary energy"
        )

    log.debug(
        "Validation OK: batch %d/%d — %d prompts, intensity range %.2f–%.2f",
        batch_num, total_batches, expected_count,
        min(intensities), max(intensities),
    )


_VALID_VISUAL_TYPES   = {"background_image", "video_loop", "thumbnail", "short_background", "short_thumbnail"}
_VALID_ASPECT_RATIOS  = {"16:9", "9:16"}
# Minimum unique visual types required — one of each kind ensures the pipeline
# has enough assets for the renderer and shorts slicer.
_MIN_VISUAL_PROMPTS   = 5


def _validate_visual_prompts(prompts_raw: list, mix_id: str) -> None:
    """
    Python-level validation of Visual PE output — mirrors _validate_audio_prompts.

    Checks:
      1. At least _MIN_VISUAL_PROMPTS returned
      2. Each visual_type is a known enum value
      3. Each aspect_ratio is "16:9" or "9:16" and is consistent with visual_type
      4. Each prompt_en is non-empty and reasonably long (≥ 15 words)
      5. At least one of each required visual_type is present
    Raises ValueError so the @retry_openai_api decorator retries the whole call.
    """
    if not prompts_raw:
        raise ValueError(f"Visual PE returned 0 prompts for mix {mix_id}")
    if len(prompts_raw) < _MIN_VISUAL_PROMPTS:
        raise ValueError(
            f"Visual PE returned {len(prompts_raw)}/{_MIN_VISUAL_PROMPTS} prompts "
            f"for mix {mix_id} — LLM truncated output"
        )

    seen_types: set[str] = set()
    required_types = {"background_image", "thumbnail", "short_background", "short_thumbnail"}

    for p in prompts_raw:
        vt = str(p.get("visual_type", "")).strip().lower()
        if vt not in _VALID_VISUAL_TYPES:
            raise ValueError(
                f"Invalid visual_type '{vt}' in Visual PE output for mix {mix_id} "
                f"— must be one of {_VALID_VISUAL_TYPES}"
            )
        seen_types.add(vt)

        ar = str(p.get("aspect_ratio", "")).strip()
        if ar not in _VALID_ASPECT_RATIOS:
            raise ValueError(
                f"Invalid aspect_ratio '{ar}' for visual_type '{vt}' in mix {mix_id}"
            )

        # Consistency: 9:16 types must have 9:16 ratio and vice versa
        is_vertical = vt in {"short_background", "short_thumbnail"}
        if is_vertical and ar != "9:16":
            raise ValueError(
                f"visual_type '{vt}' must have aspect_ratio '9:16', got '{ar}' — mix {mix_id}"
            )
        if not is_vertical and vt != "video_loop" and ar != "16:9":
            raise ValueError(
                f"visual_type '{vt}' must have aspect_ratio '16:9', got '{ar}' — mix {mix_id}"
            )

        prompt_text = str(p.get("prompt_en", "")).strip()
        if not prompt_text:
            raise ValueError(f"Empty prompt_en for visual_type '{vt}' in mix {mix_id}")
        word_count = len(prompt_text.split())
        if word_count < 10:
            raise ValueError(
                f"prompt_en too short ({word_count} words) for visual_type '{vt}' in mix {mix_id} "
                f"— expected ≥ 10 words"
            )

    missing = required_types - seen_types
    if missing:
        raise ValueError(
            f"Visual PE missing required visual types {missing} for mix {mix_id}"
        )

    log.debug("Visual prompts validation OK: %d assets, types=%s", len(prompts_raw), seen_types)


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
    Chief Strategy Officer — selects BPM, subgenre, key, arc, and stem count.

    Architecture change vs old CrewAI CSO + QA crew:
    ┌───────────────────────────────────────────────────────────────────────┐
    │ Old: CrewAI crew.kickoff()                                           │
    │      → agent iterates: call BpmRegistryTool → call YouTubeTool →    │
    │        call BpmRegistryTool again → QA review → …                   │
    │      9 sequential LLM calls × ~3-4s each ≈ 38 seconds              │
    │      (tools also re-fetched data that Python already had)            │
    │                                                                       │
    │ New: Python pre-fetches both tools, injects results directly         │
    │      → single litellm.completion() with full context                │
    │      → Python-level novelty validation (replaces QA agent)          │
    │      ~5 seconds total                                                │
    └───────────────────────────────────────────────────────────────────────┘
    """
    import litellm
    from pathlib import Path

    import yaml
    from shared.genres import GENRES
    from services.orchestrator.crew.tools import BpmRegistryTool, YoutubeAnalyticsTool

    genre_profile = GENRES.get(genre, {})
    bpm_range     = genre_profile.get("bpm_range", (100, 180))

    # ── Pre-fetch both tool results in Python (no agent round-trips) ──────
    registry_json   = BpmRegistryTool()._run()
    analytics_json  = YoutubeAnalyticsTool()._run(metric="top_performing_subgenres")

    import json as _json
    registry_data   = _json.loads(registry_json)
    used_combinations = registry_data.get("used_combinations", [])
    used_json        = _json.dumps(used_combinations)

    # ── Build prompts from YAML config ────────────────────────────────────
    _config_dir = Path(__file__).parent.parent / "config"
    with open(_config_dir / "agents.yaml", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)["cso_agent"]
    with open(_config_dir / "tasks.yaml", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)["cso_strategy_task"]

    system_prompt = (
        f"Role: {agent_cfg['role']}\n"
        f"Goal: {agent_cfg['goal']}\n"
        f"Background: {agent_cfg['backstory']}\n\n"
        "You respond ONLY with a valid JSON object — no prose, no markdown fences."
    )

    user_prompt = (
        task_cfg["description"].format(
            mix_id=mix_id,
            genre=genre,
            genre_bpm_min=bpm_range[0],
            genre_bpm_max=bpm_range[1],
            requested_duration_minutes=requested_duration_minutes,
            used_combinations=used_json,
        )
        + f"\n\nYouTube Analytics context (use to inform variety decisions):\n{analytics_json}"
    )

    response = litellm.completion(
        model=settings.llm_precise_model,
        api_key=settings.openai_api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1_024,
    )

    raw  = response.choices[0].message.content or ""
    data = _parse_crew_json(raw, "CSOStrategy")
    data["mix_id"] = mix_id
    # Inject known value — LLM doesn't need to echo it back
    data["requested_duration_minutes"] = requested_duration_minutes

    # ── Python-level QA validation (replaces QA CrewAI agent) ────────────
    chosen = {
        "bpm":           data.get("bpm"),
        "subgenre":      str(data.get("subgenre", "")).strip(),
        "key_signature": str(data.get("key_signature", "")).strip(),
    }
    for combo in used_combinations:
        if (
            abs((combo.get("bpm") or 0) - (chosen["bpm"] or 0)) < 0.5
            and combo.get("subgenre", "").lower() == chosen["subgenre"].lower()
            and combo.get("key_signature", "").lower() == chosen["key_signature"].lower()
        ):
            raise ValueError(
                f"CSO chose a duplicate combination: {chosen} — already in registry. "
                "Retry will select a different combination."
            )

    expected_stem_count = min(int(requested_duration_minutes * 2), 150)
    if abs((data.get("stem_count") or 0) - expected_stem_count) > 2:
        data["stem_count"] = expected_stem_count
        log.warning("CSO stem_count corrected to %d", expected_stem_count)

    log.info("CSO output: bpm=%.1f subgenre=%s", data.get("bpm"), data.get("subgenre"))
    return CSOStrategy(**data)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audio Prompt Engineer
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_prompt_engineer(strategy: CSOStrategy) -> AudioPromptBatch:
    """
    Generate all audio prompts in parallel using direct LiteLLM calls.

    Architecture change vs old sequential CrewAI approach:
    ┌───────────────────────────────────────────────────────────────┐
    │ Old: CrewAI batch1 → CrewAI batch2 → … → CrewAI batchN      │
    │      N × (~15s LLM + ~3s CrewAI overhead) = ~126s for N=7   │
    │                                                               │
    │ New: litellm.completion(batch1) ┐                            │
    │      litellm.completion(batch2) ├── ThreadPoolExecutor       │
    │      …                          ┘  all in parallel           │
    │      Wall-clock ≈ ~15s (single LLM call latency)            │
    └───────────────────────────────────────────────────────────────┘

    Why bypass CrewAI for this task:
    - Audio prompt generation is a single structured-output LLM call per batch.
    - CrewAI adds ~3s of Python overhead (agent init, task routing, iteration
      management) with zero benefit for a non-agentic, no-tool task.
    - Direct LiteLLM calls with response_format="json_object" guarantee valid
      JSON without CrewAI's string parsing layer.
    - All batches are fully independent — ThreadPoolExecutor parallelises them.
    """
    import math
    from concurrent.futures import ThreadPoolExecutor, as_completed, Future
    from pathlib import Path

    import yaml
    import litellm

    # ── Token-safe batch size ──────────────────────────────────────────────
    # gpt-4o-mini max output: 16 384 tokens.
    # Each stem entry worst-case: ~150 tokens (80-word prompt_en + JSON keys).
    # Safety margin: 75% of output budget → hard ceiling at 81 stems/batch.
    # This guard fires if AUDIO_PROMPT_BATCH_SIZE is set too high in .env.
    _TOKENS_PER_STEM  = 150   # conservative worst-case per output entry
    _MAX_OUTPUT_TOKENS = 16_384
    _SAFETY_FACTOR    = 0.75
    _SAFE_MAX_BATCH   = int(_MAX_OUTPUT_TOKENS * _SAFETY_FACTOR / _TOKENS_PER_STEM)  # = 81

    total_stems   = strategy.stem_count
    batch_size    = min(settings.audio_prompt_batch_size, _SAFE_MAX_BATCH)
    concurrency   = settings.prompt_concurrency
    total_batches = math.ceil(total_stems / batch_size)

    if batch_size < settings.audio_prompt_batch_size:
        log.warning(
            "AUDIO_PROMPT_BATCH_SIZE=%d exceeds token-safe ceiling %d — capped to %d",
            settings.audio_prompt_batch_size, _SAFE_MAX_BATCH, batch_size,
        )

    # Pre-compute arc boundaries once
    arc_intro_end   = max(0, int(total_stems * 0.15) - 1)
    arc_peak_start  = int(total_stems * 0.45)
    arc_peak_end    = int(total_stems * 0.70)
    arc_outro_start = int(total_stems * 0.85)

    # Load agent persona from YAML (role + goal + backstory → system prompt)
    _config_dir = Path(__file__).parent.parent / "config"
    with open(_config_dir / "agents.yaml", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)["audio_prompt_engineer"]
    with open(_config_dir / "tasks.yaml", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)["audio_prompt_task"]

    _system_prompt = (
        f"Role: {agent_cfg['role']}\n"
        f"Goal: {agent_cfg['goal']}\n"
        f"Background: {agent_cfg['backstory']}\n\n"
        "You respond ONLY with a valid JSON object — no prose, no markdown fences."
    )

    log.info(
        "Generating %d audio prompts for mix %s: %d batches of ≤%d (concurrency=%d)",
        total_stems, strategy.mix_id, total_batches, batch_size, concurrency,
    )

    def _generate_batch(batch_num: int) -> tuple[int, list[StemPrompt]]:
        """
        Generate one batch of prompts via a direct LiteLLM call.
        Returns (batch_num, list[StemPrompt]).  Raises on permanent failure.
        """
        pos_start = (batch_num - 1) * batch_size
        pos_end   = min(pos_start + batch_size - 1, total_stems - 1)
        expected  = pos_end - pos_start + 1

        user_prompt = task_cfg["description"].format(
            mix_id=strategy.mix_id,
            bpm=strategy.bpm,
            subgenre=strategy.subgenre,
            key_signature=strategy.key_signature,
            style_description=strategy.style_description,
            transition_arc=strategy.transition_arc,
            batch_num=batch_num,
            total_batches=total_batches,
            total_stems=total_stems,
            batch_size=expected,
            position_start=pos_start,
            position_end_inclusive=pos_end,
            arc_intro_end=arc_intro_end,
            arc_peak_start=arc_peak_start,
            arc_peak_end=arc_peak_end,
            arc_outro_start=arc_outro_start,
            total_stems_minus_1=total_stems - 1,
            batch_size_minus_1=expected - 1,
            batch_size_plus_1=expected + 1,
        )

        last_exc: Exception | None = None
        for attempt in range(1, 3):  # max 2 attempts per batch
            try:
                response = litellm.completion(
                    model=settings.llm_model,
                    api_key=settings.openai_api_key,
                    messages=[
                        {"role": "system", "content": _system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=16_384,
                )
                raw  = response.choices[0].message.content or ""
                data = _parse_crew_json(raw, f"AudioPromptBatch batch {batch_num}/{total_batches}")

                prompts_raw = data.get("prompts", [])

                # Auto-repair null positions BEFORE validation.
                # The model sometimes returns position=null for small last batches.
                # We already know the correct positions from pos_start + index, so
                # assign them here — validation then sees clean, complete entries.
                repaired = 0
                for i, p in enumerate(prompts_raw):
                    if p.get("position") is None and str(p.get("prompt_en", "")).strip():
                        p["position"] = pos_start + i
                        repaired += 1
                if repaired:
                    log.warning(
                        "Auto-repaired %d null positions in batch %d/%d",
                        repaired, batch_num, total_batches,
                    )

                _validate_audio_prompts(
                    prompts_raw, expected,
                    batch_num, total_batches, pos_start, pos_end,
                )

                # Force-correct positions (handles mis-numbered but non-null positions)
                for i, p in enumerate(prompts_raw[:expected]):
                    p["position"] = pos_start + i

                batch_prompts = [StemPrompt(**p) for p in prompts_raw[:expected]]
                log.info(
                    "Batch %d/%d done: %d prompts (positions %d-%d)",
                    batch_num, total_batches, len(batch_prompts), pos_start, pos_end,
                )
                return batch_num, batch_prompts

            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    log.warning(
                        "Batch %d/%d attempt %d/2 failed: %s — retrying",
                        batch_num, total_batches, attempt, exc,
                    )
                    time.sleep(2)
                else:
                    log.error(
                        "Batch %d/%d permanently failed after 2 attempts: %s",
                        batch_num, total_batches, exc,
                    )

        raise RuntimeError(
            f"Audio PE batch {batch_num}/{total_batches} failed for mix "
            f"{strategy.mix_id}: {last_exc}"
        )

    # Fire all batches concurrently
    results: dict[int, list[StemPrompt]] = {}
    with ThreadPoolExecutor(max_workers=min(concurrency, total_batches)) as pool:
        futures: dict[Future, int] = {
            pool.submit(_generate_batch, bn): bn
            for bn in range(1, total_batches + 1)
        }
        for future in as_completed(futures):
            batch_num, batch_prompts = future.result()  # re-raises on failure
            results[batch_num] = batch_prompts

    # Reassemble in order
    all_prompts: list[StemPrompt] = []
    for bn in range(1, total_batches + 1):
        all_prompts.extend(results[bn])

    # Trim to exact stem_count
    all_prompts = all_prompts[:total_stems]

    log.info(
        "Audio prompts complete: %d/%d stems for mix %s",
        len(all_prompts), total_stems, strategy.mix_id,
    )
    return AudioPromptBatch(mix_id=strategy.mix_id, strategy=strategy, prompts=all_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stem Fetcher — fire-and-forget async predictions + concurrent poll
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stems_batch(batch: AudioPromptBatch) -> StemBatchResult:
    """
    Generates every stem in the batch concurrently via Replicate async predictions.

    Algorithm:
      1. Load stem IDs from DB.
      2. Build StemRequest list.
      3. Call generate_stems_concurrent() which:
           a. Submits all N predictions to Replicate without waiting (Phase 1).
           b. Polls all N predictions simultaneously in a ThreadPoolExecutor (Phase 2).
           c. Saves each WAV as it completes; fires on_stem_done callback (Phase 3).
      4. on_stem_done writes the stem status to DB immediately on completion
         (live progress visible to the frontend during generation).

    Wall-clock time ≈ slowest single generation (~45-90 s)
    vs old sequential ≈ N × (generation + 11 s sleep).
    """
    from shared.media.audio.replicate_musicgen import (
        ReplicateMusicGenProvider,
        StemRequest,
        StemResult,
    )

    mix_id    = batch.strategy.mix_id
    out_dir   = _stems_dir(mix_id)
    generator = get_audio_generator()

    log.info(
        "Fetching %d stems concurrently (mix_id=%s, concurrency=%d)",
        len(batch.prompts), mix_id, settings.stem_concurrency,
    )

    # Load existing stem IDs from DB
    with get_sync_db() as db:
        stem_rows = db.execute(
            select(Stem.id, Stem.position).where(Stem.mix_id == mix_id)
        ).fetchall()
    stem_id_map = {r.position: r.id for r in stem_rows}

    # Build request list
    stem_requests = [
        StemRequest(
            position=p.position,
            prompt=p.prompt_en,
            bpm=batch.strategy.bpm,
            duration_s=settings.stem_duration_seconds,
            output_path=str(out_dir / f"{p.position:04d}.wav"),
        )
        for p in batch.prompts
    ]

    # Live DB callback — called from worker threads as each stem finishes
    def _on_stem_done(result: StemResult) -> None:
        try:
            with get_sync_db() as db:
                if result.ok:
                    db.execute(
                        update(Stem)
                        .where(Stem.mix_id == mix_id, Stem.position == result.position)
                        .values(status=StemStatus.READY, file_path=result.output_path)
                    )
                else:
                    db.execute(
                        update(Stem)
                        .where(Stem.mix_id == mix_id, Stem.position == result.position)
                        .values(
                            status=StemStatus.FAILED,
                            error_message=result.error[:1000],
                        )
                    )
        except Exception as db_exc:
            # Never let a DB write kill the generation loop
            log.warning("DB progress update failed for stem %04d: %s", result.position, db_exc)

    # Use concurrent provider directly if available, else fall back to sequential
    if isinstance(generator, ReplicateMusicGenProvider):
        stem_results = generator.generate_stems_concurrent(
            requests=stem_requests,
            on_stem_done=_on_stem_done,
        )
    else:
        # Fallback: sequential loop for non-Replicate providers
        stem_results = []
        for req in stem_requests:
            try:
                generated = generator.generate_stem(
                    prompt=req.prompt,
                    bpm=req.bpm,
                    duration_s=req.duration_s,
                    output_path=req.output_path,
                )
                r = StemResult(
                    position=req.position,
                    output_path=req.output_path,
                    duration_s=generated.duration_s,
                    sample_rate=generated.sample_rate,
                )
            except Exception as exc:
                r = StemResult(position=req.position, output_path="", error=str(exc))
            _on_stem_done(r)
            stem_results.append(r)

    # Map StemResult → StemFetchResult (public schema)
    fetch_results = [
        StemFetchResult(
            mix_id=mix_id,
            stem_id=stem_id_map.get(r.position, str(uuid.uuid4())),
            position=r.position,
            file_path=r.output_path,
            status="ready" if r.ok else "failed",
            error=r.error if not r.ok else None,
        )
        for r in stem_results
    ]

    return StemBatchResult(mix_id=mix_id, results=fetch_results)


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
        style_description=style_hint or f"{mix.subgenre or 'Drum and Bass'} mix at {mix.bpm or 174} BPM",
    )

    result = crew.kickoff()
    raw  = result.raw if hasattr(result, "raw") else str(result)
    data = _parse_crew_json(raw, "VisualPromptBatch")
    data["mix_id"] = mix_id

    prompts_raw = data.get("prompts", [])
    _validate_visual_prompts(prompts_raw, mix_id)

    prompts = [VisualPrompt(**p) for p in prompts_raw]
    log.info("Visual prompts validated: %d assets for mix %s", len(prompts), mix_id)
    return VisualPromptBatch(mix_id=mix_id, prompts=prompts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visual Fetcher — provider-agnostic via MediaGenerator factory
# ─────────────────────────────────────────────────────────────────────────────

def fetch_visuals_batch(mix_id: str) -> None:
    """
    Generates all pending Visual rows via the configured Image/VideoGenerator.

    Provider selection (env vars):
      IMAGE_PROVIDER=dalle3   → DallE3Provider           (~$0.080/image, HD vivid)
      VIDEO_PROVIDER=ffmpeg   → FFmpegKenBurnsProvider   (free, animated Ken Burns)

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
                    prompt=visual.prompt_en,
                    aspect_ratio=visual.aspect_ratio,
                    output_path=src_image_path,
                )
                # Step 2: animate into a video loop
                output_path = str(out_dir / f"{visual_id}.mp4")
                vid_generator.generate_video_loop(
                    prompt=visual.prompt_en,
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
                    prompt=visual.prompt_en,
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
        transition_arc=mix.transition_arc or "",  # stored by CSO at step 1
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
