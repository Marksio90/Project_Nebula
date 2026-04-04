"""
services/dsp_worker/audio/beat_aligner.py
─────────────────────────────────────────────────────────────────────────────
Beat detection and phase-alignment engine using librosa.

Core algorithm:
  1. Load stem as stereo float32 at native or target SR (44100 Hz).
  2. Compute onset strength envelope and detect beats, anchored to the CSO's
     target BPM using `librosa.beat.beat_track` with `start_bpm` + `tightness`.
  3. Find the "one" of each bar: the downbeat is the first beat in every
     4-beat group whose onset strength is maximal (strongest transient).
  4. Compute the sample-accurate trim offset so the stem's first downbeat
     aligns to sample 0 when concatenated.
  5. Return a BeatAlignedStem carrying the audio array plus all metadata
     needed for the stitcher and DB persistence.

Crossfade strategy:
  - Crossfades happen ON the downbeat ("one") of the NEXT stem.
  - Fade length = exactly 1 beat period (60 / BPM seconds).
  - Window: raised-cosine (Hann half-window) for click-free transitions.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

log = logging.getLogger("nebula.dsp.beat_aligner")

TARGET_SR  = 44_100   # All audio normalised to this sample rate
HOP_LENGTH = 512      # librosa hop length (samples)
BARS_IN_4_4 = 4       # Beats per bar — assume 4/4 throughout


@dataclass
class BeatAlignedStem:
    """
    Container for a beat-analysed audio stem.
    audio: float32 stereo array, shape (2, N_samples)
    """
    position:            int
    file_path:           str
    audio:               np.ndarray   # (2, N) float32 stereo
    sr:                  int
    bpm_detected:        float
    beat_samples:        np.ndarray   # 1-D array of beat sample positions
    downbeat_samples:    np.ndarray   # Subset: downbeat positions (bar 'one')
    beat_period_samples: int          # Samples per beat at target BPM
    bar_period_samples:  int          # Samples per bar (4 beats)
    trim_offset_samples: int          # Samples trimmed from head for alignment
    rms_db:              float


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_analyse_stem(
    file_path: str,
    target_bpm: float,
    position: int,
) -> BeatAlignedStem:
    """
    Load a 30-second WAV stem and return a fully-analysed BeatAlignedStem.

    Parameters
    ----------
    file_path   : Path to the WAV file on disk.
    target_bpm  : The CSO-selected BPM for the mix. librosa is anchored to
                  this value to correct MusicGen's slight BPM drift.
    position    : 0-indexed stem position in the mix (for logging).
    """
    log.debug("Analysing stem %04d: %s", position, file_path)

    # ── Load stereo audio ─────────────────────────────────────────────────
    audio_stereo, sr = sf.read(file_path, dtype="float32", always_2d=True)
    audio_stereo = audio_stereo.T   # → (channels, samples)

    if sr != TARGET_SR:
        log.debug("Resampling stem %04d: %d Hz → %d Hz", position, sr, TARGET_SR)
        audio_stereo = np.stack([
            librosa.resample(ch, orig_sr=sr, target_sr=TARGET_SR)
            for ch in audio_stereo
        ])
        sr = TARGET_SR

    # Ensure stereo
    if audio_stereo.shape[0] == 1:
        audio_stereo = np.vstack([audio_stereo, audio_stereo])

    y_mono = librosa.to_mono(audio_stereo)

    # ── Beat tracking — anchored tightly to target BPM ────────────────────
    tempo_raw, beat_frames = librosa.beat.beat_track(
        y=y_mono,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=target_bpm,
        tightness=120.0,    # High value forces adherence to start_bpm
        trim=False,
    )
    # librosa may return array-wrapped scalar; extract float
    bpm_detected = float(np.atleast_1d(tempo_raw)[0])

    beat_samples = librosa.frames_to_samples(beat_frames, hop_length=HOP_LENGTH)
    if len(beat_samples) == 0:
        # Fallback: synthesise beat grid from target BPM
        beat_period  = int(sr * 60.0 / target_bpm)
        beat_samples = np.arange(0, audio_stereo.shape[1], beat_period, dtype=int)
        log.warning("Stem %04d: librosa found no beats — using synthetic grid at %.1f BPM",
                    position, target_bpm)
        bpm_detected = target_bpm

    # ── Determine beat and bar periods ────────────────────────────────────
    beat_period_samples = int(round(sr * 60.0 / target_bpm))
    bar_period_samples  = beat_period_samples * BARS_IN_4_4

    # ── Onset strength for downbeat detection ─────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=HOP_LENGTH)

    downbeat_samples = _find_downbeats(
        beat_samples=beat_samples,
        onset_env=onset_env,
        hop_length=HOP_LENGTH,
        audio_len=audio_stereo.shape[1],
    )

    # ── Trim to first downbeat for phase alignment ─────────────────────────
    trim_offset = int(downbeat_samples[0]) if len(downbeat_samples) > 0 else 0
    # Never trim more than 2 bars (avoid gutting the stem)
    max_trim = 2 * bar_period_samples
    trim_offset = min(trim_offset, max_trim)

    audio_trimmed = audio_stereo[:, trim_offset:]

    # Adjust beat/downbeat positions after trim
    beat_samples     = beat_samples[beat_samples >= trim_offset] - trim_offset
    downbeat_samples = downbeat_samples[downbeat_samples >= trim_offset] - trim_offset

    # ── RMS level ─────────────────────────────────────────────────────────
    rms_linear = float(np.sqrt(np.mean(librosa.to_mono(audio_trimmed) ** 2)))
    rms_db     = float(20.0 * np.log10(max(rms_linear, 1e-9)))

    log.debug(
        "Stem %04d: bpm=%.1f beats=%d downbeats=%d trim=%d rms=%.1f dB",
        position, bpm_detected, len(beat_samples), len(downbeat_samples),
        trim_offset, rms_db,
    )

    return BeatAlignedStem(
        position=position,
        file_path=file_path,
        audio=audio_trimmed,
        sr=sr,
        bpm_detected=bpm_detected,
        beat_samples=beat_samples,
        downbeat_samples=downbeat_samples,
        beat_period_samples=beat_period_samples,
        bar_period_samples=bar_period_samples,
        trim_offset_samples=trim_offset,
        rms_db=rms_db,
    )


def compute_crossfade_join(
    stem_a: BeatAlignedStem,
    stem_b: BeatAlignedStem,
    crossfade_beats: int = 1,
) -> np.ndarray:
    """
    Produce the crossfaded overlap region between the end of stem_a and
    the start of stem_b.

    The crossfade is aligned to the DOWNBEAT:
      - stem_a fades out from its last downbeat
      - stem_b fades in from its first downbeat

    Returns a (2, fade_samples) float32 array representing the overlap mix.
    """
    fade_samples = stem_a.beat_period_samples * crossfade_beats

    # Tail of stem_a: last `fade_samples` samples
    tail = stem_a.audio[:, -fade_samples:].copy()
    if tail.shape[1] < fade_samples:
        # Pad with zeros if stem is shorter than expected
        tail = np.pad(tail, ((0, 0), (fade_samples - tail.shape[1], 0)))

    # Head of stem_b: first `fade_samples` samples
    head = stem_b.audio[:, :fade_samples].copy()
    if head.shape[1] < fade_samples:
        head = np.pad(head, ((0, 0), (0, fade_samples - head.shape[1])))

    # Raised-cosine (Hann half-window) — perceptually smooth, no click artefacts
    t         = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float32)
    fade_out  = np.cos(t) ** 2   # 1.0 → 0.0
    fade_in   = np.sin(t) ** 2   # 0.0 → 1.0

    blended = tail * fade_out[np.newaxis, :] + head * fade_in[np.newaxis, :]
    return blended.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _find_downbeats(
    beat_samples:  np.ndarray,
    onset_env:     np.ndarray,
    hop_length:    int,
    audio_len:     int,
) -> np.ndarray:
    """
    Identify bar downbeats ("the one") from a beat grid.

    Strategy:
      1. Group beats into bars of 4.
      2. For each group, find which beat has the highest onset strength.
      3. If the max-onset beat is the first in the group (i.e., beat 0 of the
         bar), that's the canonical downbeat.
      4. Otherwise, shift the grouping until beat 0 aligns with the strongest
         onset — this corrects off-by-one downbeat placement.

    Returns an array of sample positions corresponding to bar downbeats.
    """
    if len(beat_samples) < BARS_IN_4_4:
        return beat_samples[:1] if len(beat_samples) > 0 else np.array([0])

    # Convert beat samples to onset-env frames for strength lookup
    beat_frames = librosa.samples_to_frames(beat_samples, hop_length=hop_length)
    beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)

    # Find the beat phase that maximises downbeat onset strength
    best_phase = 0
    best_score = -np.inf
    for phase in range(BARS_IN_4_4):
        # Downbeat candidates with this phase offset
        candidates = beat_frames[phase::BARS_IN_4_4]
        score = float(np.mean(onset_env[candidates]))
        if score > best_score:
            best_score = score
            best_phase = phase

    downbeat_indices = np.arange(best_phase, len(beat_samples), BARS_IN_4_4)
    downbeat_samples = beat_samples[downbeat_indices]

    # Ensure the very first downbeat is ≥ 0
    downbeat_samples = downbeat_samples[downbeat_samples >= 0]
    if len(downbeat_samples) == 0:
        return np.array([0])

    return downbeat_samples
