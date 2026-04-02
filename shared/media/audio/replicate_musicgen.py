"""
shared/media/audio/replicate_musicgen.py
─────────────────────────────────────────────────────────────────────────────
AudioGenerator backed by Meta's MusicGen Stereo Large via Replicate API.

Model:  facebook/musicgen-stereo-large
Cost:   ~$0.008 per 30-second stem (as of Q1 2026)
Docs:   https://replicate.com/facebook/musicgen-stereo-large

Why MusicGen Stereo Large:
  - Stereo output — essential for a professional DJ mix
  - Publicly available, stable API, no waitlist
  - Excellent prompt following for genre/BPM instructions
  - 30-second outputs match Lyria's spec exactly
  - ~3–8s generation time per stem
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import soundfile as sf
import time

from shared.media.base import AudioGenerator, GeneratedAudio
from shared.utils.retry import retry_http

log = logging.getLogger("nebula.media.audio.replicate")

# facebook/musicgen-stereo-large was deprecated on Replicate.
# The current model is facebook/musicgen; the variant is selected via the
# model_version input parameter.
MUSICGEN_MODEL   = "facebook/musicgen"
TARGET_SR        = 44_100
DEFAULT_DURATION = 30

# Replicate free-tier / low-credit accounts are limited to 6 predictions/min
# (burst=1).  We wait this many seconds between requests to stay under the
# limit.  At paid tier (>$5 balance) the limit is much higher but the sleep
# is short enough to be harmless.
_REPLICATE_REQUEST_DELAY_S = 11


class ReplicateMusicGenProvider(AudioGenerator):
    """
    Generates 30-second stereo music stems via the Replicate API.
    Uses the `replicate` Python SDK for the API call and httpx for
    the file download — both wrapped with tenacity retries.
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required for ReplicateMusicGenProvider")
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self._api_token = api_token

    @property
    def provider_name(self) -> str:
        return "replicate/musicgen-stereo-large"

    def generate_stem(
        self,
        prompt: str,
        bpm: float,
        duration_s: int = DEFAULT_DURATION,
        output_path: str = "",
    ) -> GeneratedAudio:
        """
        Call MusicGen Stereo Large and write the result as WAV to output_path.

        The BPM is prepended to the prompt so the model is anchored to the
        correct tempo — MusicGen responds well to explicit BPM instructions.
        """
        import replicate

        # Prepend BPM anchor — improves tempo accuracy significantly
        full_prompt = f"{bpm:.0f} BPM. {prompt}"

        log.debug("MusicGen request: bpm=%.1f duration=%ds prompt='%.80s'",
                  bpm, duration_s, full_prompt)

        output = replicate.run(
            MUSICGEN_MODEL,
            input={
                "prompt":                 full_prompt,
                "model_version":          "stereo-large",
                "output_format":          "wav",
                "normalization_strategy": "peak",
                "duration":               duration_s,
                # Classifier-free guidance — higher = more prompt-adherent
                "classifier_free_guidance": 3,
                "temperature":            1.0,
                "top_k":                  250,
                "top_p":                  0.0,
            },
        )

        # Throttle to stay within Replicate's per-minute prediction limit.
        # Low-credit accounts (<$5) are capped at 6/min (burst=1) — without
        # this sleep every stem after the first would get a 429.
        time.sleep(_REPLICATE_REQUEST_DELAY_S)

        # The SDK returns either a FileOutput object or a URL string
        audio_bytes = self._extract_bytes(output)

        if not audio_bytes:
            raise RuntimeError(f"MusicGen returned empty output for prompt: {full_prompt[:80]}")

        # Write to disk
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)

        # Verify the written file is readable and get its duration
        audio_data, sr = sf.read(output_path)
        actual_duration = len(audio_data) / sr

        log.debug("MusicGen: wrote %.1f KB, %.1f s @ %d Hz → %s",
                  len(audio_bytes) / 1024, actual_duration, sr, output_path)

        return GeneratedAudio(
            file_path=output_path,
            duration_s=actual_duration,
            sample_rate=sr,
        )

    @retry_http
    def _extract_bytes(self, output) -> bytes:
        """
        Extract raw bytes from the Replicate SDK output.

        The SDK may return:
          - A FileOutput object with a .read() method
          - A URL string (older SDK versions or some models)
          - A list containing one of the above
        """
        # Unwrap list (some models return list of outputs)
        if isinstance(output, list):
            output = output[0]

        # FileOutput object (replicate SDK >= 0.25)
        if hasattr(output, "read"):
            return output.read()

        # URL string — download directly
        if isinstance(output, str) and output.startswith("http"):
            resp = httpx.get(output, timeout=120.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

        raise TypeError(f"Unexpected MusicGen output type: {type(output)}")
