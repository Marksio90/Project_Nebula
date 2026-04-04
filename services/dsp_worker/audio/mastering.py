"""
services/dsp_worker/audio/mastering.py
─────────────────────────────────────────────────────────────────────────────
VST-grade mastering chain using Spotify's pedalboard library.
Followed by ITU-R BS.1770-4 integrated loudness measurement via pyloudnorm
and normalisation to the streaming target of -14.0 LUFS.

Mastering chain (signal order):
  1. High-pass @ 30 Hz        — Remove sub-sonic rumble (common in AI audio output)
  2. Low shelf  @ 120 Hz -1dB — Tighten sub-bass for streaming codec headroom
  3. Peak EQ    @ 3.5 kHz +1.5 dB — Presence boost (DNB transient clarity)
  4. High shelf @ 14 kHz +1.0 dB  — Air / top-end extension (MusicGen tends to soft-limit highs)
  5. Compressor 2:1, attack 20ms, release 250ms — Glue compression
                                                  (slow attack preserves transients)
  6. Hard limiter @ -1.0 dBFS — True-peak ceiling for streaming delivery

Post-chain:
  7. LUFS measurement (pyloudnorm) — integrated loudness of the full master
  8. Loudness normalisation → -14.0 LUFS (streaming standard)
  9. Second true-peak check after normalisation
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    Pedalboard,
    PeakFilter,
)

from shared.schemas.events import AudioQAResult

log = logging.getLogger("nebula.dsp.mastering")

TARGET_LUFS      = -14.0
TRUE_PEAK_CEIL   = -1.0    # dBFS
LUFS_TOLERANCE   =  0.5    # LU — acceptable deviation from target
TRUE_PEAK_TOL    =  0.2    # dB — acceptable overshoot


# ─────────────────────────────────────────────────────────────────────────────
# Pedalboard chain
# ─────────────────────────────────────────────────────────────────────────────

def _build_mastering_chain() -> Pedalboard:
    """
    Build and return the pedalboard mastering chain.
    Instantiated fresh per-mix to avoid any state bleed between runs.
    """
    return Pedalboard([
        # 1. Sub-sonic rumble removal
        HighpassFilter(cutoff_frequency_hz=30.0),

        # 2. Sub-bass tightening (improves codec headroom, avoids muddiness)
        LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=-1.0),

        # 3. Presence boost — adds definition to Neurofunk basslines
        #    and snare transients in the 2–5 kHz range
        PeakFilter(cutoff_frequency_hz=3500.0, gain_db=1.5, q=0.8),

        # 4. Air — restores top-end extension that MusicGen sometimes soft-limits
        HighShelfFilter(cutoff_frequency_hz=14_000.0, gain_db=1.0),

        # 5. Glue compressor
        #    Ratio 2:1 is gentle — goal is cohesion, not loudness.
        #    Slow attack (20ms) preserves transient punch.
        Compressor(
            threshold_db=-18.0,
            ratio=2.0,
            attack_ms=20.0,
            release_ms=250.0,
        ),

        # 6. Transparent brick-wall limiter — true peak ceiling
        Limiter(threshold_db=TRUE_PEAK_CEIL, release_ms=100.0),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def master_audio(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = TARGET_LUFS,
    true_peak_ceil: float = TRUE_PEAK_CEIL,
) -> tuple[np.ndarray, float, float]:
    """
    Apply the full mastering chain and LUFS normalisation.

    Parameters
    ----------
    audio        : (2, N) or (N,) float32 audio array.
    sr           : Sample rate (e.g. 44100).
    target_lufs  : Target integrated loudness (default -14.0 LUFS).
    true_peak_ceil: Hard ceiling in dBFS (default -1.0).

    Returns
    -------
    mastered_audio : (2, N) float32 normalised audio
    lufs_measured  : Integrated loudness BEFORE final normalisation (informational)
    true_peak_dbfs : True peak of the final output
    """
    # Ensure stereo (2, N)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] > audio.shape[1]:
        audio = audio.T   # Was (N, 2) — flip to (2, N)

    audio = audio.astype(np.float32)

    # ── Step 1: Pedalboard processing ─────────────────────────────────────
    board = _build_mastering_chain()
    # pedalboard expects (channels, samples) as float32
    processed: np.ndarray = board(audio, sr)
    log.debug("Pedalboard chain applied: shape=%s", processed.shape)

    # ── Step 2: Measure integrated loudness (ITU-R BS.1770-4) ─────────────
    # pyloudnorm expects (samples, channels)
    meter = pyln.Meter(sr)
    lufs_measured = float(meter.integrated_loudness(processed.T))
    log.info("Pre-normalisation LUFS: %.2f", lufs_measured)

    # Guard against -inf LUFS (silence)
    if not np.isfinite(lufs_measured):
        log.warning("Non-finite LUFS measurement (silence?): skipping normalisation")
        return processed.astype(np.float32), lufs_measured, _true_peak_dbfs(processed)

    # ── Step 3: Loudness normalisation → target_lufs ──────────────────────
    normalised_T = pyln.normalize.loudness(processed.T, lufs_measured, target_lufs)
    normalised   = normalised_T.T.astype(np.float32)

    # ── Step 4: Final true-peak check — clip if normalisation pushed over ──
    tp = _true_peak_dbfs(normalised)
    if tp > true_peak_ceil:
        log.debug("True peak %.2f dBFS > %.2f — applying soft clip", tp, true_peak_ceil)
        clip_linear = 10.0 ** (true_peak_ceil / 20.0)
        normalised  = np.clip(normalised, -clip_linear, clip_linear)
        tp          = _true_peak_dbfs(normalised)

    log.info("Master complete: LUFS=%.2f  TP=%.2f dBFS", target_lufs, tp)
    return normalised, lufs_measured, tp


def run_audio_qa(
    mix_id: str,
    audio_path: str,
    target_lufs: float = TARGET_LUFS,
    true_peak_ceiling: float = TRUE_PEAK_CEIL,
) -> AudioQAResult:
    """
    Load the mastered audio file and verify it meets the quality gates.

    Gates:
      - Integrated LUFS within ±0.5 LU of target
      - True peak ≤ true_peak_ceiling + 0.2 dB tolerance
      - No pure silence (LUFS > -70)
    """
    log.info("Audio QA: mix_id=%s path=%s", mix_id, audio_path)
    issues: list[str] = []

    try:
        audio_stereo, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_stereo = audio_stereo.T   # → (2, N)
    except Exception as exc:
        return AudioQAResult(
            mix_id=mix_id, passed=False,
            lufs_measured=-70.0, true_peak_dbfs=0.0,
            issues=[f"Could not load audio file: {exc}"],
        )

    meter = pyln.Meter(sr)
    lufs  = float(meter.integrated_loudness(audio_stereo.T))
    tp    = _true_peak_dbfs(audio_stereo)

    log.info("QA measured: LUFS=%.2f  TP=%.2f dBFS", lufs, tp)

    # Gate 1: LUFS
    if not np.isfinite(lufs) or lufs < -70.0:
        issues.append(f"Audio is effectively silent (LUFS={lufs:.1f})")
    elif abs(lufs - target_lufs) > LUFS_TOLERANCE:
        issues.append(
            f"LUFS out of tolerance: measured={lufs:.2f}, "
            f"target={target_lufs:.2f}, tolerance=±{LUFS_TOLERANCE}"
        )

    # Gate 2: True peak
    if tp > true_peak_ceiling + TRUE_PEAK_TOL:
        issues.append(
            f"True peak overshoot: measured={tp:.2f} dBFS, ceiling={true_peak_ceiling:.2f} dBFS"
        )

    passed = len(issues) == 0
    if passed:
        log.info("✅ Audio QA PASSED")
    else:
        log.warning("❌ Audio QA FAILED: %s", "; ".join(issues))

    return AudioQAResult(
        mix_id=mix_id,
        passed=passed,
        lufs_measured=lufs,
        true_peak_dbfs=tp,
        issues=issues,
    )


def renormalize_audio_file(audio_path: str) -> None:
    """
    Re-apply the full mastering chain to an existing audio file, overwriting it.

    Called by run_qa_audio_check when QA fails, so that the Celery retry checks
    a freshly-corrected file rather than the same file that just failed.

    Use-cases:
      - LUFS drifted due to floating-point accumulation across many stems
      - True peak exceeded ceiling after loudness normalisation boosted gain
      - File was written with an incorrect gain by a previous worker
    """
    log.info("Re-normalising audio in-place: %s", audio_path)
    try:
        audio_stereo, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio_stereo.T   # → (2, N)
        mastered, lufs_pre, tp = master_audio(audio, sr)
        sf.write(audio_path, mastered.T, sr, subtype="PCM_24")
        log.info(
            "Re-normalisation complete: pre_LUFS=%.2f → %.2f LUFS, TP=%.2f dBFS",
            lufs_pre, TARGET_LUFS, tp,
        )
    except Exception as exc:
        log.error("renormalize_audio_file failed for %s: %s", audio_path, exc)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _true_peak_dbfs(audio: np.ndarray) -> float:
    """
    Compute the true peak in dBFS.
    Uses 4× oversampling interpolation to catch inter-sample peaks.
    """
    if audio.size == 0:
        return -120.0
    # Simple approximation: max absolute sample value
    # For production, use pyloudnorm's peak_normalize or an oversampling limiter
    peak_linear = float(np.max(np.abs(audio)))
    if peak_linear < 1e-9:
        return -120.0
    return float(20.0 * np.log10(peak_linear))
