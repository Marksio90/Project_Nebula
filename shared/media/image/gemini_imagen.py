"""
shared/media/image/gemini_imagen.py
─────────────────────────────────────────────────────────────────────────────
ImageGenerator backed by Google Imagen 3 via the google-genai SDK.

STATUS: Generally available — activate with IMAGE_PROVIDER=gemini in .env.
        Default is DALL-E 3 because it uses the same already-configured key.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

from shared.media.base import GeneratedImage, ImageGenerator
from shared.utils.retry import retry_gemini_api

log = logging.getLogger("nebula.media.image.gemini_imagen")

_AR_MAP = {"16:9": "16:9", "9:16": "9:16", "1:1": "1:1"}


class GeminiImagenProvider(ImageGenerator):

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini/imagen-3"

    @retry_gemini_api
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str,
        output_path: str,
    ) -> GeneratedImage:
        from google import genai as google_genai
        from google.genai import types

        client = google_genai.Client(api_key=self._api_key)
        api_ar = _AR_MAP.get(aspect_ratio, "16:9")

        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=api_ar,
                safety_filter_level="block_only_high",
            ),
        )

        if not response.generated_images:
            raise ValueError("Imagen 3 returned no images")

        image_bytes = response.generated_images[0].image.image_bytes
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(image_bytes)

        # Determine approximate dimensions from aspect ratio
        dims = {"16:9": (1792, 1024), "9:16": (1024, 1792)}.get(aspect_ratio, (1024, 1024))
        return GeneratedImage(
            file_path=output_path,
            width=dims[0], height=dims[1],
            aspect_ratio=aspect_ratio,
        )
