"""
services/video_worker/video/slicer.py
─────────────────────────────────────────────────────────────────────────────
Viral Content Slicer — finds the 3 loudest 60-second drops and renders
each as a 9:16 1080×1920 vertical video for YouTube Shorts and TikTok.

Algorithm:
  1. Load the mastered audio with librosa and compute a sliding-window RMS
     energy curve (60-second windows, 5-second hop).
  2. Select the 3 highest-RMS, non-overlapping windows (minimum 60s apart).
  3. For each selected window:
       a. Slice the corresponding segment from the full mix video (FFmpeg).
       b. Reframe from 16:9 → 9:16 using a centre-crop + scale pipeline.
       c. Add the 9:16 short background or a vertical crop of the mix video.
       d. Overlay the waveform visualiser (vertical orientation).
  4. Export each short to EXPORTS_DIR/{mix_id}/short_{rank}.mp4 (1080×1920).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sqlalchemy import select

from services.video_worker.video.renderer import _run_ffmpeg, _select_codec
from shared.config import get_settings
from shared.db.models import Mix, Visual, VisualStatus, VisualType
from shared.db.session import get_sync_db
from shared.schemas.events import ViralShortResult, ViralSliceResult

log      = logging.getLogger("nebula.video.slicer")
settings = get_settings()

SHORT_DURATION  = 60.0    # seconds
SHORT_WIDTH     = 1080    # 9:16 vertical width
SHORT_HEIGHT    = 1920    # 9:16 vertical height
RMS_WINDOW_S    = 60.0    # sliding window size for RMS analysis
RMS_HOP_S       =  5.0    # hop size (resolution of RMS curve)
MIN_GAP_S       = 60.0    # minimum separation between selected windows (non-overlap)
TOP_N_SHORTS    =  3      # number of viral shorts to produce


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def slice_viral_shorts_from_mix(mix_id: str) -> ViralSliceResult:
    """
    Find the 3 highest-energy 60-second segments and render them as 9:16 shorts.
    """
    log.info("▶ slice_viral_shorts: mix_id=%s", mix_id)

    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()
        visuals_9_16 = db.execute(
            select(Visual).where(
                Visual.mix_id == mix_id,
                Visual.aspect_ratio == "9:16",
                Visual.status == VisualStatus.READY,
            )
        ).scalars().all()

    if not mix.mastered_audio_path or not Path(mix.mastered_audio_path).exists():
        raise FileNotFoundError(f"Mastered audio missing: {mix.mastered_audio_path}")

    if not mix.full_video_path or not Path(mix.full_video_path).exists():
        raise FileNotFoundError(f"Full mix video missing: {mix.full_video_path}")

    # ── 1. RMS analysis ────────────────────────────────────────────────────
    log.info("Analysing RMS energy curve: %s", mix.mastered_audio_path)
    rms_windows = _compute_rms_windows(
        audio_path=mix.mastered_audio_path,
        window_s=RMS_WINDOW_S,
        hop_s=RMS_HOP_S,
    )
    log.info("RMS windows computed: %d candidates", len(rms_windows))

    # ── 2. Select top 3 non-overlapping windows ────────────────────────────
    selected = _select_top_non_overlapping(
        rms_windows=rms_windows,
        top_n=TOP_N_SHORTS,
        min_gap_s=MIN_GAP_S,
    )
    log.info("Selected drops: %s", [(f"{s:.0f}s", f"{r:.1f}dB") for s, r in selected])

    # ── 3. Render each short ───────────────────────────────────────────────
    out_dir = Path(settings.exports_dir) / mix_id
    out_dir.mkdir(parents=True, exist_ok=True)

    codec, codec_args = _select_codec()
    results: list[ViralShortResult] = []

    for rank, (start_s, rms_db) in enumerate(selected, start=1):
        short_id   = str(uuid.uuid4())
        video_path = str(out_dir / f"short_{rank}.mp4")

        # Select a 9:16 background (fall back to centre-crop of the 16:9 mix)
        bg_9_16 = _pick_9_16_background(visuals_9_16)

        try:
            _render_short(
                mix_video_path=mix.full_video_path,
                audio_path=mix.mastered_audio_path,
                bg_9_16_path=bg_9_16,
                start_s=start_s,
                duration_s=SHORT_DURATION,
                output_path=video_path,
                codec=codec,
                codec_args=codec_args,
                mix_id=mix_id,
                rank=rank,
            )
            results.append(ViralShortResult(
                mix_id=mix_id,
                short_id=short_id,
                rank=rank,
                start_seconds=round(start_s, 2),
                rms_db=round(rms_db, 2),
                video_path=video_path,
            ))
            log.info("✅ Short #%d rendered: start=%.0fs rms=%.1fdB → %s",
                     rank, start_s, rms_db, video_path)

        except Exception as exc:
            log.error("Short #%d FAILED: %s", rank, exc)
            # Non-fatal — include a failed entry so the pipeline can continue
            results.append(ViralShortResult(
                mix_id=mix_id,
                short_id=short_id,
                rank=rank,
                start_seconds=round(start_s, 2),
                rms_db=round(rms_db, 2),
                video_path="",
            ))

    return ViralSliceResult(mix_id=mix_id, shorts=results)


# ─────────────────────────────────────────────────────────────────────────────
# RMS analysis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rms_windows(
    audio_path: str,
    window_s: float,
    hop_s: float,
) -> list[tuple[float, float]]:
    """
    Compute sliding-window RMS (in dBFS) over the mastered audio.

    Returns a list of (start_seconds, rms_db) tuples, sorted by rms_db descending.
    """
    # Load as mono at reduced SR for speed (22050 Hz is enough for RMS)
    y, sr = librosa.load(audio_path, sr=22_050, mono=True)

    window_samples = int(window_s * sr)
    hop_samples    = int(hop_s    * sr)
    n_samples      = len(y)

    windows: list[tuple[float, float]] = []

    for start in range(0, n_samples - window_samples, hop_samples):
        segment   = y[start : start + window_samples]
        rms_lin   = float(np.sqrt(np.mean(segment ** 2)))
        rms_db    = float(20.0 * np.log10(max(rms_lin, 1e-9)))
        start_sec = start / sr
        windows.append((start_sec, rms_db))

    # Sort by RMS descending — loudest windows first
    windows.sort(key=lambda x: x[1], reverse=True)
    return windows


def _select_top_non_overlapping(
    rms_windows: list[tuple[float, float]],
    top_n: int,
    min_gap_s: float,
) -> list[tuple[float, float]]:
    """
    Greedily select the top_n highest-RMS windows that are at least
    min_gap_s seconds apart from each other.

    This guarantees that each viral short highlights a distinct musical moment.
    """
    selected: list[tuple[float, float]] = []

    for start_s, rms_db in rms_windows:
        # Check against already-selected windows
        overlaps = any(abs(start_s - sel_s) < min_gap_s for sel_s, _ in selected)
        if not overlaps:
            selected.append((start_s, rms_db))
        if len(selected) == top_n:
            break

    # Sort selected windows chronologically for upload ordering
    selected.sort(key=lambda x: x[0])
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Short renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_short(
    mix_video_path: str,
    audio_path: str,
    bg_9_16_path: str | None,
    start_s: float,
    duration_s: float,
    output_path: str,
    codec: str,
    codec_args: list[str],
    mix_id: str,
    rank: int,
) -> None:
    """
    Render a single 9:16 viral short via FFmpeg.

    Filtergraph:
      - Take the 16:9 mix video, centre-crop to 9:16 ratio
      - OR composite over a dedicated 9:16 background
      - Overlay a vertical waveform visualiser on the right edge
      - Burn-in the audio segment from the mastered mix
    """
    w, h = SHORT_WIDTH, SHORT_HEIGHT

    # ── Inputs ────────────────────────────────────────────────────────────
    input_args: list[str] = []

    # Input 0: audio slice
    input_args += [
        "-ss", str(start_s),
        "-t",  str(duration_s),
        "-i",  audio_path,
    ]

    # Input 1: background (9:16 visual or centre-cropped mix video)
    if bg_9_16_path and Path(bg_9_16_path).exists():
        input_args += [
            "-ss", str(start_s),
            "-t",  str(duration_s),
            "-loop", "1" if bg_9_16_path.endswith(".png") else "0",
            "-i",  bg_9_16_path,
        ]
        bg_chain = f"[1:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1[bg];"
    else:
        # Centre-crop the 16:9 mix video to 9:16
        # 16:9 at width W → crop vertical strip of width W*(9/16) centred
        crop_w = int(h * 9 / 16)   # Width of the 9:16 crop from a 1080-tall source
        input_args += [
            "-ss", str(start_s),
            "-t",  str(duration_s),
            "-i",  mix_video_path,
        ]
        bg_chain = (
            f"[1:v]scale=-1:{h},crop={w}:{h}:(in_w-{w})/2:0,"
            f"setsar=1[bg];"
        )

    # ── Filtergraph ───────────────────────────────────────────────────────
    # Vertical waveform on the right edge (200px wide)
    vis_w = 200
    fg = (
        f"{bg_chain}"
        f"[0:a]showwaves="
        f"s={vis_w}x{h}:"
        f"mode=cline:"
        f"colors=0x00E5FF@0.9:"
        f"scale=lin:"
        f"draw=full[vis];"
        f"[bg][vis]overlay={w - vis_w}:0[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", fg,
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", codec, *codec_args,
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-t", str(duration_s),
        "-movflags", "+faststart",
        output_path,
    ]

    _run_ffmpeg(cmd, mix_id=mix_id, label=f"short_{rank}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_9_16_background(visuals: list) -> str | None:
    """Pick the first available READY 9:16 background image or video."""
    for v in visuals:
        if v.visual_type in (VisualType.SHORT_BACKGROUND, VisualType.VIDEO_LOOP):
            if v.file_path and Path(v.file_path).exists():
                return v.file_path
    return None
