"""
shared/media/video/ffmpeg_kenburns.py
─────────────────────────────────────────────────────────────────────────────
VideoGenerator that creates loopable video clips from static images using
FFmpeg's `zoompan` filter — the Ken Burns effect.

Why FFmpeg Ken Burns (not a cloud video model):
  - Zero API cost — pure FFmpeg, runs locally inside the Docker container
  - Deterministic — same image + seed → same video every time
  - No rate limits, no waitlists, works fully offline
  - Produces polished, professional results that loop seamlessly
  - Full control over duration, speed, and animation style

Ken Burns animations available (randomly selected per asset):
  1. Slow zoom in from centre          (classic cinematic)
  2. Slow zoom out + drift right       (revealing effect)
  3. Slow pan left → right             (landscape sweep)
  4. Slow pan right → left             (reverse sweep)
  5. Zoom in + gentle tilt down        (descending feel — dark DnB aesthetic)

Output: H.264 MP4, 30fps, seamlessly loopable (first frame ≈ last frame).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from shared.media.base import GeneratedVideo, VideoGenerator

log = logging.getLogger("nebula.media.video.kenburns")

FPS      = 30
DURATION = 8     # seconds per loop — feels natural with DnB energy

# Zoom/pan parameters
_ZOOM_START = 1.0
_ZOOM_END   = 1.35   # 35% zoom over DURATION seconds → subtle but cinematic
_ZOOM_STEP  = (_ZOOM_END - _ZOOM_START) / (FPS * DURATION)

# Resolution targets
_RES_MAP = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}

# Pre-scale factor — zoompan needs a larger canvas to pan/zoom into
_SCALE_FACTOR = 2.2


def _zoom_in_centre(w: int, h: int, d: int) -> str:
    """Classic Ken Burns: slow zoom into the image centre."""
    return (
        f"scale={int(w*_SCALE_FACTOR)}:{int(h*_SCALE_FACTOR)},"
        f"zoompan="
        f"z='min(zoom+{_ZOOM_STEP:.6f},{_ZOOM_END})':"
        f"d={d}:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)',"
        f"scale={w}:{h},setsar=1"
    )


def _zoom_out_drift_right(w: int, h: int, d: int) -> str:
    """Start zoomed in, slowly pull back while drifting right."""
    return (
        f"scale={int(w*_SCALE_FACTOR)}:{int(h*_SCALE_FACTOR)},"
        f"zoompan="
        f"z='if(lte(zoom,{_ZOOM_START}),{_ZOOM_END},max({_ZOOM_START+0.001:.3f},zoom-{_ZOOM_STEP:.6f}))':"
        f"d={d}:"
        f"x='min(iw-iw/zoom, (iw-iw/zoom)*on/{d})':"
        f"y='ih/2-(ih/zoom/2)',"
        f"scale={w}:{h},setsar=1"
    )


def _pan_left_to_right(w: int, h: int, d: int) -> str:
    """Hold zoom, sweep the frame left to right."""
    zoom = 1.2
    return (
        f"scale={int(w*_SCALE_FACTOR)}:{int(h*_SCALE_FACTOR)},"
        f"zoompan="
        f"z={zoom}:"
        f"d={d}:"
        f"x='(iw-iw/zoom)*on/{d}':"
        f"y='ih/2-(ih/zoom/2)',"
        f"scale={w}:{h},setsar=1"
    )


def _pan_right_to_left(w: int, h: int, d: int) -> str:
    """Hold zoom, sweep frame right to left — mirror of above."""
    zoom = 1.2
    return (
        f"scale={int(w*_SCALE_FACTOR)}:{int(h*_SCALE_FACTOR)},"
        f"zoompan="
        f"z={zoom}:"
        f"d={d}:"
        f"x='(iw-iw/zoom)*(1-on/{d})':"
        f"y='ih/2-(ih/zoom/2)',"
        f"scale={w}:{h},setsar=1"
    )


def _zoom_in_tilt_down(w: int, h: int, d: int) -> str:
    """Zoom in while tilting the frame downward — ominous, dark DnB feel."""
    return (
        f"scale={int(w*_SCALE_FACTOR)}:{int(h*_SCALE_FACTOR)},"
        f"zoompan="
        f"z='min(zoom+{_ZOOM_STEP:.6f},{_ZOOM_END})':"
        f"d={d}:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='min(ih-ih/zoom, (ih-ih/zoom)*on/{d})',"
        f"scale={w}:{h},setsar=1"
    )


# All animation styles indexed 0–4
_STYLES = [
    _zoom_in_centre,
    _zoom_out_drift_right,
    _pan_left_to_right,
    _pan_right_to_left,
    _zoom_in_tilt_down,
]


class FFmpegKenBurnsProvider(VideoGenerator):
    """
    Animates a static image into a loopable video using FFmpeg zoompan.
    Selects animation style deterministically from the image content hash
    so the same image always produces the same animation.
    """

    @property
    def provider_name(self) -> str:
        return "ffmpeg/kenburns"

    def generate_video_loop(
        self,
        prompt: str,                       # unused — animation is image-driven
        aspect_ratio: str,
        output_path: str,
        source_image_path: str | None = None,
    ) -> GeneratedVideo:
        if not source_image_path or not Path(source_image_path).exists():
            raise ValueError(
                f"FFmpegKenBurnsProvider requires a source_image_path. "
                f"Got: {source_image_path}"
            )

        w, h  = _RES_MAP.get(aspect_ratio, (1920, 1080))
        d     = FPS * DURATION    # total frames

        # Deterministic style selection based on image file hash
        img_hash  = hashlib.md5(Path(source_image_path).read_bytes()).hexdigest()
        style_idx = int(img_hash[:4], 16) % len(_STYLES)
        vf_chain  = _STYLES[style_idx](w, h, d)

        log.debug(
            "Ken Burns: style=%d size=%dx%d duration=%ds → %s",
            style_idx, w, h, DURATION, output_path
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i",    source_image_path,
            "-vf",   vf_chain,
            "-t",    str(DURATION),
            "-r",    str(FPS),
            "-c:v",  "libx264",
            "-preset", "fast",
            "-crf",  "20",
            "-pix_fmt", "yuv420p",    # Maximum compatibility
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg Ken Burns failed (exit {result.returncode}):\n{result.stderr[-1000:]}"
            )

        log.info(
            "✅ Ken Burns: style=%d %dx%d %ds → %s",
            style_idx, w, h, DURATION, output_path
        )

        return GeneratedVideo(
            file_path=output_path,
            duration_s=float(DURATION),
            width=w,
            height=h,
            source="kenburns",
        )
