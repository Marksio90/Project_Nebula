"""
shared/media/audio/replicate_musicgen.py
─────────────────────────────────────────────────────────────────────────────
AudioGenerator backed by Meta's MusicGen via Replicate API.

Model selection (in priority order):
  1. REPLICATE_MODEL env var (default: "facebook/musicgen-stereo-large")
  2. Auto-fallback to other known MusicGen slugs if the primary 404s

Why version pinning matters:
  replicate.run("owner/model") requires a "deployment" on Replicate.
  replicate.run("owner/model:sha256...") always works if the version exists.
  This provider automatically resolves the latest version hash on startup.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import soundfile as sf
import time

from shared.config import get_settings
from shared.media.base import AudioGenerator, GeneratedAudio
from shared.utils.retry import retry_http

log = logging.getLogger("nebula.media.audio.replicate")

TARGET_SR        = 44_100
DEFAULT_DURATION = 30

# Replicate free-tier: ~6 predictions/min. 11s sleep keeps us safely under.
_REPLICATE_REQUEST_DELAY_S = 11

# Fallback slugs tried in order if the primary REPLICATE_MODEL 404s.
# The versioned slug below was auto-discovered 2026-04-04 and is the current
# working MusicGen model on Replicate. If it ever 404s again, the provider
# will re-discover automatically via _resolve_model().
_CANDIDATE_SLUGS = [
    "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
    "meta/musicgen",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen",
]


class ReplicateMusicGenProvider(AudioGenerator):
    """
    Generates 30-second stereo music stems via the Replicate API.
    Auto-resolves the model slug to a pinned version hash on first use
    so 404 "model not found" errors are caught at startup, not mid-pipeline.
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required for ReplicateMusicGenProvider")
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self._api_token = api_token
        configured = get_settings().replicate_model
        self._model = self._resolve_model(configured)
        log.info("ReplicateMusicGenProvider ready: model=%s", self._model)

    # ── Model resolution ──────────────────────────────────────────────────────

    def _resolve_model(self, configured_slug: str) -> str:
        """
        Resolve a model slug to a pinned "owner/model:version_hash" reference.

        Strategy:
          1. If the slug already contains ":" it is already versioned — use as-is.
          2. Otherwise query the Replicate API for the model's latest_version.id.
          3. If the configured slug 404s, try each slug in _CANDIDATE_SLUGS.
          4. If nothing works, return the slug as-is (will fail loudly on first use).
        """
        import replicate as _r

        # Build the ordered list: configured slug first, then fallbacks
        candidates = [configured_slug] + [s for s in _CANDIDATE_SLUGS if s != configured_slug]

        for slug in candidates:
            # Already versioned — trust the operator
            if ":" in slug:
                log.info("Model slug already versioned: %s", slug)
                return slug

            try:
                model_obj = _r.models.get(slug)
                latest = model_obj.latest_version
                if latest:
                    versioned = f"{slug}:{latest.id}"
                    log.info("Resolved %s → %s", slug, versioned)
                    return versioned
                else:
                    # Model exists but has no public versions — try it unversioned
                    log.warning("Model %s has no public version — using slug", slug)
                    return slug
            except Exception as exc:
                log.warning("Model %s unavailable (%s) — trying next candidate", slug, exc)
                continue

        log.error(
            "No working Replicate model found. Tried: %s\n"
            "Fix: set REPLICATE_MODEL in .env to a valid slug from "
            "https://replicate.com/explore?category=audio",
            candidates,
        )
        return configured_slug  # Will raise on first prediction with a clear message

    # ── Stem generation ───────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return f"replicate/{self._model.split('/')[1].split(':')[0]}"

    def generate_stem(
        self,
        prompt: str,
        bpm: float,
        duration_s: int = DEFAULT_DURATION,
        output_path: str = "",
    ) -> GeneratedAudio:
        """
        Call MusicGen and write the result as WAV to output_path.
        BPM is prepended to the prompt — MusicGen responds well to explicit tempo cues.
        """
        import replicate

        full_prompt = f"{bpm:.0f} BPM. {prompt}"

        log.debug("MusicGen request: bpm=%.1f duration=%ds prompt='%.80s'",
                  bpm, duration_s, full_prompt)

        try:
            output = replicate.run(
                self._model,
                input={
                    "prompt":                   full_prompt,
                    "output_format":            "wav",
                    "normalization_strategy":   "peak",
                    "duration":                 duration_s,
                    "classifier_free_guidance": 3,
                    "temperature":              1.0,
                    "top_k":                    250,
                    "top_p":                    0.0,
                },
            )
        finally:
            # Throttle even on failure to avoid rate-limit cascade on the next stem
            time.sleep(_REPLICATE_REQUEST_DELAY_S)

        audio_bytes = self._extract_bytes(output)

        if not audio_bytes:
            raise RuntimeError(f"MusicGen returned empty output for prompt: {full_prompt[:80]}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)

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
        Extract raw audio bytes from the Replicate SDK output.
        Handles FileOutput objects (SDK ≥ 0.25), URL strings, and lists.
        """
        if isinstance(output, list):
            output = output[0]

        if hasattr(output, "read"):
            return output.read()

        if isinstance(output, str) and output.startswith("http"):
            resp = httpx.get(output, timeout=120.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

        raise TypeError(f"Unexpected MusicGen output type: {type(output)}")
