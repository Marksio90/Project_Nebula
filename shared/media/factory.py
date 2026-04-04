"""
shared/media/factory.py
─────────────────────────────────────────────────────────────────────────────
Provider factory — returns the correct AudioGenerator, ImageGenerator, or
VideoGenerator instance based on the AUDIO_PROVIDER / IMAGE_PROVIDER /
VIDEO_PROVIDER environment variables.

Production stack (zero waitlists, zero extra setup):
  AUDIO_PROVIDER=replicate   → ReplicateMusicGenProvider  (~$0.008/stem)
  IMAGE_PROVIDER=dalle3      → DallE3Provider              (~$0.080/image HD)
  VIDEO_PROVIDER=ffmpeg      → FFmpegKenBurnsProvider      (FREE, local CPU)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging

from shared.config import get_settings
from shared.media.base import AudioGenerator, ImageGenerator, VideoGenerator

log = logging.getLogger("nebula.media.factory")


def get_audio_generator() -> AudioGenerator:
    """Return the configured AudioGenerator. AUDIO_PROVIDER=replicate (default)."""
    settings = get_settings()
    provider = settings.audio_provider.lower()

    if provider == "replicate":
        from shared.media.audio.replicate_musicgen import ReplicateMusicGenProvider
        log.debug("Audio provider: replicate/musicgen")
        return ReplicateMusicGenProvider(api_token=settings.replicate_api_token)

    raise ValueError(
        f"Unknown AUDIO_PROVIDER={provider!r}. Valid options: replicate"
    )


def get_image_generator() -> ImageGenerator:
    """Return the configured ImageGenerator. IMAGE_PROVIDER=dalle3 (default)."""
    settings = get_settings()
    provider = settings.image_provider.lower()

    if provider == "dalle3":
        from shared.media.image.dalle3 import DallE3Provider
        log.debug("Image provider: openai/dall-e-3")
        return DallE3Provider(api_key=settings.openai_api_key)

    raise ValueError(
        f"Unknown IMAGE_PROVIDER={provider!r}. Valid options: dalle3"
    )


def get_video_generator() -> VideoGenerator:
    """Return the configured VideoGenerator. VIDEO_PROVIDER=ffmpeg (default)."""
    settings = get_settings()
    provider = settings.video_provider.lower()

    if provider == "ffmpeg":
        from shared.media.video.ffmpeg_kenburns import FFmpegKenBurnsProvider
        log.debug("Video provider: ffmpeg/kenburns (local, zero cost)")
        return FFmpegKenBurnsProvider()

    raise ValueError(
        f"Unknown VIDEO_PROVIDER={provider!r}. Valid options: ffmpeg"
    )
