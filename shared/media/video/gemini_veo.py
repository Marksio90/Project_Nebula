"""
shared/media/video/gemini_veo.py
─────────────────────────────────────────────────────────────────────────────
VideoGenerator backed by Google Veo 2 via the google-genai SDK.

STATUS: Experimental / waitlist access required (Q1 2026).
        This provider is kept for future activation when Veo 2 reaches GA.
        Set VIDEO_PROVIDER=gemini in .env to enable.

To activate: ensure your Gemini API key has Veo 2 access and set:
  VIDEO_PROVIDER=gemini
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from shared.media.base import GeneratedVideo, VideoGenerator
from shared.utils.retry import retry_gemini_api

log = logging.getLogger("nebula.media.video.gemini_veo")

# Veo 2 output dimensions per aspect ratio
_DIMS_MAP = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}

_VEO_POLL_INTERVAL = 10   # seconds between status polls
_VEO_POLL_TIMEOUT  = 600  # max wait: 10 minutes


class GeminiVeoProvider(VideoGenerator):
    """
    Generates short video loops via Google Veo 2.
    Requires whitelist/GA access to the veo-2.0-generate-001 model.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini/veo-2"

    @retry_gemini_api
    def generate_video_loop(
        self,
        prompt: str,
        aspect_ratio: str,
        output_path: str,
        source_image_path: str | None = None,
    ) -> GeneratedVideo:
        from google import genai as google_genai
        from google.genai import types

        client = google_genai.Client(api_key=self._api_key)
        w, h   = _DIMS_MAP.get(aspect_ratio, (1920, 1080))

        generate_kwargs: dict = dict(
            model="veo-2.0-generate-001",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=8,
                number_of_videos=1,
            ),
        )

        # Image-conditioned generation if a source image is provided
        if source_image_path and Path(source_image_path).exists():
            image_bytes = Path(source_image_path).read_bytes()
            generate_kwargs["image"] = types.Image(
                image_bytes=image_bytes,
                mime_type="image/png",
            )
            log.debug("Veo 2: image-conditioned generation from %s", source_image_path)

        operation = client.models.generate_videos(**generate_kwargs)

        # Poll until the async operation completes
        elapsed = 0
        while not operation.done:
            if elapsed >= _VEO_POLL_TIMEOUT:
                raise TimeoutError(
                    f"Veo 2 operation timed out after {_VEO_POLL_TIMEOUT}s"
                )
            time.sleep(_VEO_POLL_INTERVAL)
            elapsed += _VEO_POLL_INTERVAL
            operation = client.operations.get(operation)
            log.debug("Veo 2: polling operation (elapsed=%ds)", elapsed)

        if not operation.response or not operation.response.generated_videos:
            raise ValueError("Veo 2 returned no video in response")

        video = operation.response.generated_videos[0]
        video_bytes = client.files.download(file=video.video)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(video_bytes)

        log.info("Veo 2: %dx%d 8s video → %s", w, h, output_path)

        return GeneratedVideo(
            file_path=output_path,
            duration_s=8.0,
            width=w,
            height=h,
            source="veo2",
        )
