"""
shared/media/audio/gemini_lyria.py
─────────────────────────────────────────────────────────────────────────────
AudioGenerator backed by Google Gemini Lyria 3.

STATUS: Experimental / waitlist access required (Q1 2026).
        This provider is kept for future activation when Lyria 3 reaches GA.
        Set AUDIO_PROVIDER=gemini in .env to enable.

To activate: ensure your Gemini API key has Lyria 3 access and set:
  AUDIO_PROVIDER=gemini
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from shared.media.base import AudioGenerator, GeneratedAudio
from shared.utils.retry import retry_gemini_api

log = logging.getLogger("nebula.media.audio.gemini_lyria")


class GeminiLyriaProvider(AudioGenerator):
    """
    Generates audio stems via Gemini Lyria 3.
    Requires whitelist/GA access to the lyria-realtime-exp model.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "gemini/lyria-3"

    @retry_gemini_api
    def generate_stem(
        self,
        prompt: str,
        bpm: float,
        duration_s: int = 30,
        output_path: str = "",
    ) -> GeneratedAudio:
        # Uses google-genai (new unified SDK) — no google-generativeai needed
        from google import genai as google_genai
        from google.genai import types

        client = google_genai.Client(api_key=self._api_key)

        full_prompt = (
            f"{prompt}\n\n"
            f"Technical parameters: {bpm:.0f} BPM, {duration_s} seconds duration, "
            f"stereo, 44100 Hz sample rate, high quality production."
        )

        response = client.models.generate_content(
            model="lyria-realtime-exp",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="audio/wav",
            ),
        )

        audio_bytes: bytes | None = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                audio_bytes = part.inline_data.data
                break

        if not audio_bytes:
            raise ValueError("Lyria 3 returned no audio data")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)

        audio_data, sr = sf.read(output_path)
        return GeneratedAudio(
            file_path=output_path,
            duration_s=len(audio_data) / sr,
            sample_rate=sr,
        )
