"""
tests/unit/test_audio/test_beat_aligner.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the beat alignment and crossfade engine.
Uses synthetically generated audio so no real stems are required.
─────────────────────────────────────────────────────────────────────────────
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from services.dsp_worker.audio.beat_aligner import (
    BeatAlignedStem,
    TARGET_SR,
    compute_crossfade_join,
    load_and_analyse_stem,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_click_track(bpm: float, duration_s: float, sr: int = TARGET_SR) -> np.ndarray:
    """
    Synthesise a click track at `bpm` BPM for `duration_s` seconds.
    Each beat is a short 800 Hz sine burst. Stereo output (2, N).
    """
    n_samples = int(duration_s * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_period = int(sr * 60.0 / bpm)
    burst_len   = int(0.010 * sr)   # 10 ms burst
    t_burst     = np.linspace(0, 2 * np.pi * 800 * 0.010, burst_len)

    for beat_start in range(0, n_samples - burst_len, beat_period):
        audio[beat_start : beat_start + burst_len] += 0.5 * np.sin(t_burst)

    stereo = np.stack([audio, audio])
    return stereo.astype(np.float32)


@pytest.fixture
def stem_wav_174(tmp_path: Path) -> str:
    """Write a 30-second 174 BPM click track to a temp WAV file."""
    audio = _make_click_track(bpm=174.0, duration_s=30.0)
    wav_path = str(tmp_path / "stem_174.wav")
    sf.write(wav_path, audio.T, TARGET_SR, subtype="PCM_16")
    return wav_path


@pytest.fixture
def stem_wav_170(tmp_path: Path) -> str:
    """Write a 30-second 170 BPM click track (slight BPM drift from target)."""
    audio = _make_click_track(bpm=170.0, duration_s=30.0)
    wav_path = str(tmp_path / "stem_170.wav")
    sf.write(wav_path, audio.T, TARGET_SR, subtype="PCM_16")
    return wav_path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_and_analyse_stem
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndAnalyseStem:
    def test_returns_beat_aligned_stem(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        assert isinstance(stem, BeatAlignedStem)
        assert stem.position == 0
        assert stem.sr == TARGET_SR

    def test_audio_is_stereo(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        assert stem.audio.ndim == 2
        assert stem.audio.shape[0] == 2   # (channels, samples)

    def test_beat_period_matches_target_bpm(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        expected_period = int(TARGET_SR * 60.0 / 174.0)
        # Allow ±2 samples tolerance for rounding
        assert abs(stem.beat_period_samples - expected_period) <= 2

    def test_downbeats_subset_of_beats(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        assert len(stem.downbeat_samples) <= len(stem.beat_samples)
        assert len(stem.downbeat_samples) >= 1

    def test_rms_db_reasonable_range(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        # A click track at 0.5 amplitude should be in a reasonable dB range
        assert -60.0 < stem.rms_db < 0.0

    def test_trim_offset_non_negative(self, stem_wav_174: str):
        stem = load_and_analyse_stem(stem_wav_174, target_bpm=174.0, position=0)
        assert stem.trim_offset_samples >= 0

    def test_bpm_drift_corrected(self, stem_wav_170: str):
        """
        A stem generated at 170 BPM, analysed with target=174 BPM.
        librosa should be nudged to 174 BPM by the tightness parameter.
        """
        stem = load_and_analyse_stem(stem_wav_170, target_bpm=174.0, position=5)
        # Beat period should be anchored close to 174 BPM, not 170
        expected_174 = int(TARGET_SR * 60.0 / 174.0)
        assert abs(stem.beat_period_samples - expected_174) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_crossfade_join
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCrossfadeJoin:
    def _make_stem(self, n_samples: int = TARGET_SR * 30) -> BeatAlignedStem:
        """Create a minimal BeatAlignedStem for crossfade testing."""
        sr  = TARGET_SR
        bpm = 174.0
        beat_period = int(sr * 60.0 / bpm)
        audio = np.random.randn(2, n_samples).astype(np.float32) * 0.3
        beats = np.arange(0, n_samples, beat_period)
        return BeatAlignedStem(
            position=0,
            file_path="/fake/path.wav",
            audio=audio,
            sr=sr,
            bpm_detected=bpm,
            beat_samples=beats,
            downbeat_samples=beats[::4],
            beat_period_samples=beat_period,
            bar_period_samples=beat_period * 4,
            trim_offset_samples=0,
            rms_db=-12.0,
        )

    def test_output_shape(self):
        stem_a = self._make_stem()
        stem_b = self._make_stem()
        blend  = compute_crossfade_join(stem_a, stem_b, crossfade_beats=1)
        expected_len = stem_a.beat_period_samples
        assert blend.shape == (2, expected_len)

    def test_output_dtype_float32(self):
        stem_a = self._make_stem()
        stem_b = self._make_stem()
        blend  = compute_crossfade_join(stem_a, stem_b, crossfade_beats=1)
        assert blend.dtype == np.float32

    def test_blend_within_amplitude_range(self):
        """Blended region should not exceed the max amplitude of either input."""
        stem_a = self._make_stem()
        stem_b = self._make_stem()
        blend  = compute_crossfade_join(stem_a, stem_b, crossfade_beats=1)
        max_input = max(np.max(np.abs(stem_a.audio)), np.max(np.abs(stem_b.audio)))
        # Allow a tiny float tolerance above max
        assert float(np.max(np.abs(blend))) <= max_input * 1.001

    def test_crossfade_starts_near_stem_a_tail(self):
        """First sample of the blend should be close to the last beat of stem_a."""
        sr   = TARGET_SR
        bpm  = 174.0
        beat = int(sr * 60.0 / bpm)
        n    = sr * 30

        # stem_a: constant 1.0 signal
        audio_a = np.ones((2, n), dtype=np.float32)
        # stem_b: constant 0.0 signal
        audio_b = np.zeros((2, n), dtype=np.float32)

        beats = np.arange(0, n, beat)
        stem_a = self._make_stem()
        stem_a = BeatAlignedStem(
            position=0, file_path="", audio=audio_a, sr=sr,
            bpm_detected=bpm, beat_samples=beats, downbeat_samples=beats[::4],
            beat_period_samples=beat, bar_period_samples=beat * 4,
            trim_offset_samples=0, rms_db=-0.1,
        )
        stem_b = BeatAlignedStem(
            position=1, file_path="", audio=audio_b, sr=sr,
            bpm_detected=bpm, beat_samples=beats, downbeat_samples=beats[::4],
            beat_period_samples=beat, bar_period_samples=beat * 4,
            trim_offset_samples=0, rms_db=-120.0,
        )

        blend = compute_crossfade_join(stem_a, stem_b, crossfade_beats=1)

        # First sample should be close to 1.0 (stem_a dominates), last close to 0.0
        assert blend[0, 0] > 0.95, f"Expected ~1.0, got {blend[0, 0]}"
        assert blend[0, -1] < 0.05, f"Expected ~0.0, got {blend[0, -1]}"
