"""
shared/media/audio/replicate_musicgen.py
─────────────────────────────────────────────────────────────────────────────
AudioGenerator backed by Meta's MusicGen via Replicate API.

Model selection (in priority order):
  1. REPLICATE_MODEL env var (default: "facebook/musicgen-stereo-large")
  2. Auto-fallback to other known MusicGen slugs if the primary 404s

Concurrency model — fire-and-forget + concurrent poll:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Old (sequential):  stem0 →[gen+wait]→ stem1 →[gen+wait]→ …       │
  │  128 stems × ~41 s = ~87 min                                       │
  │                                                                     │
  │  New (concurrent):  submit all → poll all in ThreadPoolExecutor    │
  │  time ≈ slowest single generation ≈ ~45–90 s                       │
  └─────────────────────────────────────────────────────────────────────┘

  generate_stem()          — single stem, blocking (used by tests / fallback)
  generate_stems_concurrent() — N stems, fire-all + concurrent poll
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import soundfile as sf

from shared.config import get_settings
from shared.media.base import AudioGenerator, GeneratedAudio
from shared.utils.retry import retry_http

log = logging.getLogger("nebula.media.audio.replicate")

TARGET_SR        = 44_100
DEFAULT_DURATION = 30

# Fallback slugs tried in order if the primary REPLICATE_MODEL 404s.
_CANDIDATE_SLUGS = [
    "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
    "meta/musicgen",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen",
]

# Max wait per prediction before declaring it timed-out (seconds).
_PREDICTION_TIMEOUT_S = 600


@dataclass
class StemRequest:
    """Input descriptor for a single stem in a concurrent batch."""
    position: int
    prompt:   str
    bpm:      float
    duration_s: int = DEFAULT_DURATION
    output_path: str = ""


@dataclass
class StemResult:
    """Outcome of a single stem generation attempt."""
    position:   int
    output_path: str
    duration_s:  float = 0.0
    sample_rate: int   = 0
    error:       str   = ""

    @property
    def ok(self) -> bool:
        return not self.error


class ReplicateMusicGenProvider(AudioGenerator):
    """
    Generates stereo music stems via the Replicate API.

    Two generation modes:
      • generate_stem()              – single stem, sequential (test / fallback)
      • generate_stems_concurrent()  – N stems, fire-all + concurrent poll
                                       ~50× faster for large batches
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required for ReplicateMusicGenProvider")
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self._api_token  = api_token
        self._settings   = get_settings()
        configured       = self._settings.replicate_model
        self._model      = self._resolve_model(configured)
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

        candidates = [configured_slug] + [s for s in _CANDIDATE_SLUGS if s != configured_slug]

        for slug in candidates:
            if ":" in slug:
                log.info("Model slug already versioned: %s", slug)
                return slug
            try:
                model_obj = _r.models.get(slug)
                latest    = model_obj.latest_version
                if latest:
                    versioned = f"{slug}:{latest.id}"
                    log.info("Resolved %s → %s", slug, versioned)
                    return versioned
                else:
                    log.warning("Model %s has no public version — using slug", slug)
                    return slug
            except Exception as exc:
                log.warning("Model %s unavailable (%s) — trying next candidate", slug, exc)

        log.error(
            "No working Replicate model found. Tried: %s\n"
            "Fix: set REPLICATE_MODEL in .env to a valid slug from "
            "https://replicate.com/explore?category=audio",
            candidates,
        )
        return configured_slug

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return f"replicate/{self._model.split('/')[1].split(':')[0]}"

    def _build_input(self, full_prompt: str, duration_s: int) -> dict:
        return {
            "prompt":                   full_prompt,
            "output_format":            "wav",
            "normalization_strategy":   "peak",
            "duration":                 duration_s,
            "classifier_free_guidance": 3,
            "temperature":              1.0,
            "top_k":                    250,
            "top_p":                    0.0,
        }

    def _full_prompt(self, prompt: str, bpm: float) -> str:
        return f"{bpm:.0f} BPM. {prompt}"

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

    def _save_audio(self, audio_bytes: bytes, output_path: str) -> GeneratedAudio:
        """Write bytes to disk and return GeneratedAudio metadata."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(audio_bytes)
        audio_data, sr = sf.read(output_path)
        duration = len(audio_data) / sr
        log.debug("Saved %.1f KB, %.1f s @ %d Hz → %s",
                  len(audio_bytes) / 1024, duration, sr, output_path)
        return GeneratedAudio(file_path=output_path, duration_s=duration, sample_rate=sr)

    # ── Single-stem blocking API (tests / fallback) ───────────────────────────

    def generate_stem(
        self,
        prompt:      str,
        bpm:         float,
        duration_s:  int  = DEFAULT_DURATION,
        output_path: str  = "",
    ) -> GeneratedAudio:
        """
        Blocking single-stem generation.  Kept for tests and simple callers.
        For production batches use generate_stems_concurrent().
        """
        import replicate

        full_prompt = self._full_prompt(prompt, bpm)
        log.debug("MusicGen request: bpm=%.1f duration=%ds prompt='%.80s'",
                  bpm, duration_s, full_prompt)

        output = replicate.run(self._model, input=self._build_input(full_prompt, duration_s))
        audio_bytes = self._extract_bytes(output)
        if not audio_bytes:
            raise RuntimeError(f"MusicGen returned empty output for prompt: {full_prompt[:80]}")
        return self._save_audio(audio_bytes, output_path)

    # ── Concurrent batch API ──────────────────────────────────────────────────

    def generate_stems_concurrent(
        self,
        requests:         list[StemRequest],
        on_stem_done:     Callable[[StemResult], None] | None = None,
        concurrency:      int   | None = None,
        submit_delay_s:   float | None = None,
    ) -> list[StemResult]:
        """
        Generate N stems with fire-and-forget async predictions + concurrent poll.

        Algorithm:
          Phase 1 — Submit:  POST each prediction to Replicate (non-blocking).
                             Waits `submit_delay_s` between submissions to
                             stay inside the API rate-limit window.
          Phase 2 — Poll:    ThreadPoolExecutor with `concurrency` workers
                             calls prediction.wait() on each.  Workers run in
                             parallel so all stems are polled simultaneously.
          Phase 3 — Save:    Each completed prediction downloads + writes WAV.

        Wall-clock time ≈ max(submission_ramp, slowest_single_generation)
        vs old sequential ≈ N × (generation + sleep)

        Args:
            requests:       Ordered list of StemRequest descriptors.
            on_stem_done:   Optional callback fired after each stem completes
                            (or fails) — use for live DB progress updates.
            concurrency:    Max simultaneous poll threads.  Defaults to
                            settings.stem_concurrency (env: STEM_CONCURRENCY).
            submit_delay_s: Seconds between prediction submissions.  Defaults
                            to settings.replicate_submit_delay_s.

        Returns:
            List[StemResult] in the same order as `requests`.
        """
        import replicate

        if not requests:
            return []

        cfg             = self._settings
        concurrency     = concurrency    or cfg.stem_concurrency
        submit_delay_s  = submit_delay_s if submit_delay_s is not None else cfg.replicate_submit_delay_s

        results: list[StemResult | None] = [None] * len(requests)

        # ── Phase 1: submit all predictions ───────────────────────────────────
        # predictions[i] = (request_index, StemRequest, prediction_object)
        submitted: list[tuple[int, StemRequest, object]] = []

        log.info(
            "Submitting %d stem predictions (submit_delay=%.2fs, concurrency=%d)",
            len(requests), submit_delay_s, concurrency,
        )

        for idx, req in enumerate(requests):
            full_prompt = self._full_prompt(req.prompt, req.bpm)
            try:
                pred = replicate.predictions.create(
                    model=self._model,
                    input=self._build_input(full_prompt, req.duration_s),
                )
                submitted.append((idx, req, pred))
                log.debug("Submitted stem %04d → prediction %s", req.position, pred.id)
            except Exception as exc:
                log.error("Failed to submit stem %04d: %s", req.position, exc)
                result = StemResult(position=req.position, output_path="", error=str(exc))
                results[idx] = result
                if on_stem_done:
                    on_stem_done(result)

            if submit_delay_s > 0 and idx < len(requests) - 1:
                time.sleep(submit_delay_s)

        log.info("All %d predictions submitted; polling for results…", len(submitted))

        # ── Phase 2 & 3: wait + download concurrently ─────────────────────────
        def _wait_and_save(args: tuple[int, StemRequest, object]) -> StemResult:
            idx, req, pred = args
            try:
                pred.wait(timeout=_PREDICTION_TIMEOUT_S)
                pred.reload()

                if pred.status == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {pred.error}")
                if pred.status != "succeeded":
                    raise RuntimeError(
                        f"Prediction ended with unexpected status '{pred.status}'"
                    )

                audio_bytes = self._extract_bytes(pred.output)
                if not audio_bytes:
                    raise RuntimeError("MusicGen returned empty output")

                saved = self._save_audio(audio_bytes, req.output_path)
                return StemResult(
                    position=req.position,
                    output_path=req.output_path,
                    duration_s=saved.duration_s,
                    sample_rate=saved.sample_rate,
                )
            except Exception as exc:
                log.error("Stem %04d FAILED (prediction %s): %s", req.position, pred.id, exc)
                return StemResult(position=req.position, output_path="", error=str(exc)[:1000])

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_idx = {
                pool.submit(_wait_and_save, item): item[0]
                for item in submitted
            }
            for future in as_completed(future_to_idx):
                idx    = future_to_idx[future]
                result = future.result()          # _wait_and_save never raises
                results[idx] = result
                log.debug(
                    "Stem %04d %s",
                    result.position,
                    "ready" if result.ok else f"FAILED: {result.error[:60]}",
                )
                if on_stem_done:
                    on_stem_done(result)

        # Fill any slots that were never submitted (shouldn't happen but be safe)
        for i, r in enumerate(results):
            if r is None:
                results[i] = StemResult(
                    position=requests[i].position,
                    output_path="",
                    error="Prediction never submitted",
                )

        ready  = sum(1 for r in results if r.ok)
        failed = len(results) - ready
        log.info("Concurrent generation done: %d ready, %d failed", ready, failed)
        return results
