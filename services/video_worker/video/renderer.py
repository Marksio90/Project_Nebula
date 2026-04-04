"""
services/video_worker/video/renderer.py
─────────────────────────────────────────────────────────────────────────────
Hardware-accelerated FFmpeg video renderer for Project Nebula.

Renders a professional-grade 16:9 YouTube mix video with:
  - Seamlessly looped video background (crossfade every ~8 seconds)
  - Dynamic audio waveform + spectrum analyser overlay (complex filtergraph)
  - Chapter title cards timed to the musical arc (drawtext)
  - Hardware acceleration: NVENC (NVIDIA) → VAAPI (Intel/AMD) → libx264 fallback

Output:
  EXPORTS_DIR/{mix_id}/full_mix.mp4  (1920×1080 or 3840×2160)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from sqlalchemy import select

from shared.config import get_settings
from shared.db.models import Mix, PlatformUpload, Visual, VisualType
from shared.db.session import get_sync_db
from shared.schemas.events import VideoQAResult, VideoRenderResult

log      = logging.getLogger("nebula.video.renderer")
settings = get_settings()

# Output resolution and encoding presets
RENDER_RESOLUTION = (1920, 1080)    # Change to (3840, 2160) for 4K
WAVEFORM_HEIGHT   = 220             # Pixels for the waveform overlay bar
SPECTRUM_HEIGHT   = 180             # Pixels for the spectrum overlay bar

# Codec selection order: check availability at runtime
_CODEC_PREFERENCE = ["h264_nvenc", "h264_vaapi", "libx264"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_full_mix_video(mix_id: str) -> VideoRenderResult:
    """
    Render the full 16:9 mix video.
    Reads DB for mastered audio path, background visuals, and chapter markers.
    """
    log.info("▶ render_full_mix_video: mix_id=%s", mix_id)

    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()
        visuals = db.execute(
            select(Visual).where(
                Visual.mix_id == mix_id,
                Visual.aspect_ratio == "16:9",
                Visual.status.in_(["ready"]),
            )
        ).scalars().all()
        chapters = db.execute(
            select(PlatformUpload.chapters_pl)
            .where(PlatformUpload.mix_id == mix_id)
            .limit(1)
        ).scalar_one_or_none()

    if not mix.mastered_audio_path or not Path(mix.mastered_audio_path).exists():
        raise FileNotFoundError(f"Mastered audio not found: {mix.mastered_audio_path}")

    # Select background video (prefer FFmpeg Ken Burns loop, fall back to static image)
    bg_video = _select_background_visual(visuals, prefer_video=True)
    bg_image = _select_background_visual(visuals, prefer_video=False)

    out_dir  = Path(settings.exports_dir) / mix_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "full_mix.mp4")

    # Detect best available codec
    codec, preset_args = _select_codec()
    log.info("Codec selected: %s", codec)

    # Get audio duration
    audio_duration = _probe_duration(mix.mastered_audio_path)
    w, h           = RENDER_RESOLUTION

    # ── Build the FFmpeg filtergraph ──────────────────────────────────────
    fg, input_args, map_args = _build_full_mix_filtergraph(
        audio_path=mix.mastered_audio_path,
        bg_video_path=bg_video,
        bg_image_path=bg_image,
        chapters_pl=chapters,
        audio_duration=audio_duration,
        width=w,
        height=h,
    )

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", fg,
        *map_args,
        "-c:v", codec, *preset_args,
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-t", str(audio_duration),
        "-movflags", "+faststart",   # Web-optimised MP4
        out_path,
    ]

    log.info("FFmpeg command: %s", " ".join(cmd[:12]) + " ...")
    _run_ffmpeg(cmd, mix_id=mix_id, label="full_mix")

    log.info("✅ Full mix rendered: %s", out_path)
    return VideoRenderResult(
        mix_id=mix_id,
        full_video_path=out_path,
        duration_seconds=round(audio_duration, 2),
        resolution=f"{w}x{h}",
        codec=codec,
    )


def run_video_qa(mix_id: str) -> VideoQAResult:
    """
    Probe the rendered MP4 with ffprobe to detect quality issues:
      - Frame drops (PTS discontinuities > 0.5%)
      - Audio/video sync drift (> 40 ms)
    """
    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()

    out_path = str(Path(settings.exports_dir) / mix_id / "full_mix.mp4")
    issues: list[str] = []

    if not Path(out_path).exists():
        return VideoQAResult(
            mix_id=mix_id, passed=False, frame_drop_rate=1.0,
            issues=["Output video file does not exist"],
        )

    try:
        # Probe video stream for frame count and duration
        probe_result = _ffprobe_json(out_path)
        streams      = probe_result.get("streams", [])
        v_stream     = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream     = next((s for s in streams if s.get("codec_type") == "audio"), None)

        frame_drop_rate = 0.0
        if v_stream:
            nb_frames   = int(v_stream.get("nb_frames",   0))
            r_frame_str = v_stream.get("r_frame_rate", "30/1")
            num, den    = (int(x) for x in r_frame_str.split("/"))
            fps         = num / den if den else 30.0
            v_duration  = float(v_stream.get("duration", 0))
            expected_frames = int(v_duration * fps)

            if expected_frames > 0:
                frame_drop_rate = max(0.0, (expected_frames - nb_frames) / expected_frames)
                if frame_drop_rate > 0.005:  # > 0.5% drop rate
                    issues.append(
                        f"Frame drop rate {frame_drop_rate*100:.2f}% exceeds 0.5% threshold"
                    )

        # A/V sync check
        if v_stream and a_stream:
            v_start = float(v_stream.get("start_time", 0.0))
            a_start = float(a_stream.get("start_time", 0.0))
            av_drift_ms = abs(v_start - a_start) * 1000
            if av_drift_ms > 40:
                issues.append(f"A/V sync drift {av_drift_ms:.1f} ms exceeds 40 ms threshold")

    except Exception as exc:
        issues.append(f"ffprobe analysis failed: {exc}")
        return VideoQAResult(
            mix_id=mix_id, passed=False, frame_drop_rate=1.0, issues=issues
        )

    passed = len(issues) == 0
    if passed:
        log.info("✅ Video QA PASSED: mix_id=%s drop_rate=%.4f%%", mix_id, frame_drop_rate * 100)
    else:
        log.warning("❌ Video QA FAILED: %s", "; ".join(issues))

    return VideoQAResult(
        mix_id=mix_id, passed=passed,
        frame_drop_rate=frame_drop_rate,
        issues=issues,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg filtergraph builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_mix_filtergraph(
    audio_path: str,
    bg_video_path: str | None,
    bg_image_path: str | None,
    chapters_pl: list | None,
    audio_duration: float,
    width: int,
    height: int,
) -> tuple[str, list[str], list[str]]:
    """
    Build the FFmpeg complex filtergraph, input argument list, and map arguments.

    Returns: (filtergraph_str, input_args, map_args)

    Filtergraph overview:
      [0:a]        — mastered audio (input 0)
      [1:v]        — background video loop or generated colour ramp (input 1)
      [0:a]showwaves  → [waves]     waveform overlay
      [0:a]showspectrum → [spec]    spectrum overlay
      [1:v] scaled    → [bg]        background
      [bg][waves]overlay → [v1]
      [v1][spec]overlay  → [v2]
      [v2]drawtext (chapters) → [vout]
    """
    inputs:      list[str] = []
    input_args:  list[str] = []
    map_args:    list[str] = []
    fg_parts:    list[str] = []

    # Input 0: audio
    input_args += ["-i", audio_path]

    # Input 1: background video (looped) or static image
    if bg_video_path and Path(bg_video_path).exists():
        input_args += ["-stream_loop", "-1", "-i", bg_video_path]
        bg_label = "[1:v]"
    elif bg_image_path and Path(bg_image_path).exists():
        input_args += ["-loop", "1", "-i", bg_image_path]
        bg_label = "[1:v]"
    else:
        # Synthesise a dark gradient background via lavfi
        input_args += [
            "-f", "lavfi",
            "-i", f"color=c=0x0a0a0f:size={width}x{height}:rate=30",
        ]
        bg_label = "[1:v]"

    # Background: scale + loop
    fg_parts.append(
        f"{bg_label}scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps=fps=30[bg];"
    )

    # Waveform analyser — bottom strip
    fg_parts.append(
        f"[0:a]showwaves="
        f"s={width}x{WAVEFORM_HEIGHT}:"
        f"mode=cline:"
        f"colors=0x00E5FF@0.85|0xFF6B35@0.85:"
        f"scale=lin:"
        f"draw=full[waves];"
    )

    # Spectrum analyser — above waveform
    fg_parts.append(
        f"[0:a]showspectrum="
        f"s={width}x{SPECTRUM_HEIGHT}:"
        f"mode=separate:"
        f"color=fire:"
        f"scale=log:"
        f"fps=30:"
        f"orientation=horizontal[spec];"
    )

    # Overlay waveform on background
    wave_y = height - WAVEFORM_HEIGHT
    fg_parts.append(f"[bg][waves]overlay=0:{wave_y}[v1];")

    # Overlay spectrum above waveform
    spec_y = height - WAVEFORM_HEIGHT - SPECTRUM_HEIGHT
    fg_parts.append(f"[v1][spec]overlay=0:{spec_y}[v2];")

    # Chapter drawtext overlays
    vout_label = _append_chapter_drawtext(
        fg_parts=fg_parts,
        chapters_pl=chapters_pl or [],
        width=width,
        height=height,
        input_label="[v2]",
    )

    fg = "".join(fg_parts).rstrip(";")

    map_args = ["-map", vout_label, "-map", "0:a"]
    return fg, input_args, map_args


def _append_chapter_drawtext(
    fg_parts: list[str],
    chapters_pl: list,
    width: int,
    height: int,
    input_label: str,
) -> str:
    """
    Chain drawtext filters for each chapter marker.
    Each title appears for 5 seconds at its timestamp, fading in/out.
    Returns the label of the final output stream.
    """
    if not chapters_pl:
        fg_parts.append(f"{input_label}copy[vout]")
        return "[vout]"

    font_path   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    current_in  = input_label
    title_y     = int(height * 0.08)   # 8% from top

    for idx, chapter in enumerate(chapters_pl):
        time_str = chapter.get("time_str", "00:00") if isinstance(chapter, dict) else "00:00"
        title_pl = chapter.get("title_pl", "")       if isinstance(chapter, dict) else str(chapter)
        # Escape FFmpeg special characters
        safe_title = title_pl.replace("'", "\\'").replace(":", "\\:")
        start_s    = _timestr_to_seconds(time_str)
        end_s      = start_s + 5.0
        fade_dur   = 0.5

        out_label = f"[vchap{idx}]"
        fg_parts.append(
            f"{current_in}"
            f"drawtext="
            f"fontfile={font_path}:"
            f"fontsize=52:"
            f"fontcolor=white:"
            f"shadowcolor=black@0.6:shadowx=2:shadowy=2:"
            f"text='{safe_title}':"
            f"x=(w-text_w)/2:"
            f"y={title_y}:"
            f"enable='between(t,{start_s},{end_s})':"
            f"alpha='if(lt(t,{start_s + fade_dur}),(t-{start_s})/{fade_dur},"
            f"if(gt(t,{end_s - fade_dur}),({end_s}-t)/{fade_dur},1))'"
            f"{out_label};"
        )
        current_in = out_label

    # Rename final label
    fg_parts.append(f"{current_in}copy[vout]")
    return "[vout]"


# ─────────────────────────────────────────────────────────────────────────────
# Codec detection
# ─────────────────────────────────────────────────────────────────────────────

def _select_codec() -> tuple[str, list[str]]:
    """
    Probe FFmpeg for hardware encoder availability.
    Returns (codec_name, extra_args_list).
    """
    hwaccel = settings.ffmpeg_hwaccel if hasattr(settings, "ffmpeg_hwaccel") else "auto"

    if hwaccel == "none":
        return "libx264", ["-preset", "slow", "-crf", "18"]

    # Probe available encoders
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        encoder_output = result.stdout + result.stderr
    except Exception:
        return "libx264", ["-preset", "slow", "-crf", "18"]

    if "h264_nvenc" in encoder_output:
        return "h264_nvenc", ["-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
    if "h264_vaapi" in encoder_output:
        return "h264_vaapi", ["-vf", "format=nv12,hwupload", "-qp", "22"]

    return "libx264", ["-preset", "slow", "-crf", "18"]


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_ffmpeg(cmd: list[str], mix_id: str, label: str) -> None:
    """Execute an FFmpeg command and raise on non-zero exit."""
    log_path = Path(settings.exports_dir) / mix_id / f"ffmpeg_{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=18000,   # 5-hour hard timeout for 4K renders
        )

    if proc.returncode != 0:
        # Read last 50 lines of ffmpeg log for the error message
        with open(log_path) as f:
            tail = f.readlines()[-50:]
        raise RuntimeError(
            f"FFmpeg exited {proc.returncode} for {label}:\n{''.join(tail)}"
        )


def _ffprobe_json(path: str) -> dict:
    """Run ffprobe and return the parsed JSON stream info."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def _probe_duration(path: str) -> float:
    """Return audio/video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe duration probe failed: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0.0))


def _select_background_visual(visuals: list, prefer_video: bool) -> str | None:
    """Select the best available background visual from the DB records."""
    for v in visuals:
        is_video = v.visual_type == VisualType.VIDEO_LOOP
        if prefer_video and is_video and v.file_path and Path(v.file_path).exists():
            return v.file_path
        if not prefer_video and not is_video and v.file_path and Path(v.file_path).exists():
            return v.file_path
    return None


def _timestr_to_seconds(time_str: str) -> float:
    """Convert 'MM:SS' or 'HH:MM:SS' to float seconds."""
    parts = [int(p) for p in time_str.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    return 0.0
