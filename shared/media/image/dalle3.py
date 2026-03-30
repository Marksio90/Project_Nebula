"""
shared/media/image/dalle3.py
─────────────────────────────────────────────────────────────────────────────
ImageGenerator backed by OpenAI DALL-E 3.

Why DALL-E 3:
  - Already have the OpenAI API key (used for CrewAI/GPT-4o)
  - Zero additional setup — no new account, no new waitlist
  - HD quality mode produces cinematic results
  - Native 16:9 (1792×1024) and 9:16 (1024×1792) support
  - Consistent, high-quality outputs with strong prompt following

Cost: ~$0.080 per HD image (16:9 or 9:16) as of Q1 2026
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from shared.media.base import GeneratedImage, ImageGenerator
from shared.utils.retry import retry_openai_api

log = logging.getLogger("nebula.media.image.dalle3")

# DALL-E 3 supported sizes
_SIZE_MAP = {
    "16:9":  "1792x1024",
    "9:16":  "1024x1792",
    "1:1":   "1024x1024",
}

_DIMS_MAP = {
    "1792x1024": (1792, 1024),
    "1024x1792": (1024, 1792),
    "1024x1024": (1024, 1024),
}


class DallE3Provider(ImageGenerator):
    """
    Generates images via DALL-E 3 HD.
    Uses the same OpenAI API key already configured for CrewAI.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai/dall-e-3"

    @retry_openai_api
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str,
        output_path: str,
    ) -> GeneratedImage:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)

        size     = _SIZE_MAP.get(aspect_ratio, "1792x1024")
        w, h     = _DIMS_MAP[size]

        log.debug("DALL-E 3 request: size=%s prompt='%.80s'", size, prompt)

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,       # type: ignore[arg-type]
            quality="hd",
            style="vivid",   # More cinematic than "natural" for music visuals
            n=1,
        )

        image_url = response.data[0].url
        if not image_url:
            raise ValueError("DALL-E 3 returned no image URL")

        # Download the image
        image_bytes = self._download(image_url)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(image_bytes)

        log.debug("DALL-E 3: wrote %.1f KB → %s", len(image_bytes) / 1024, output_path)

        return GeneratedImage(
            file_path=output_path,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
        )

    @retry_http
    def _download(self, url: str) -> bytes:
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
