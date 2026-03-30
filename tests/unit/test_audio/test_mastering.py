"""
tests/unit/test_audio/test_mastering.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the pedalboard mastering chain and LUFS QA.
─────────────────────────────────────────────────────────────────────────────
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from services.dsp_worker.audio.mastering import (
    TARGET_LUFS,
    TRUE_PEAK_CEIL,
    LUFS_TOLERANCE,
    master_audio,
    run_audio_qa,
)


SR = 44_100


def _make_pink_noise(duration_s: float = 10.0, lufs_target: float = -20.0) -> np.ndarray:
    """Generate approximate pink noise normalised to lufs_target."""
    import pyloudnorm as pyln
    n = int(duration_s * SR)
    # White noise → apply 1/f shaping (rough approximation)
    white  = np.random.randn(n).astype(np.float32)
    freqs  = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    pink_spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink   = np.fft.irfft(pink_spectrum, n=n).astype(np.float32)
    stereo = np.stack([pink, pink])

    meter   = pyln.Meter(SR)
    current = meter.integrated_loudness(stereo.T)
    if np.isfinite(current):
        import pyloudnorm as pyln2
        normalised = pyln2.normalize.loudness(stereo.T, current, lufs_target).T
        return normalised.astype(np.float32)
    return stereo * 0.1


class TestMasterAudio:
    def test_output_shape_preserved(self):
        audio = _make_pink_noise(duration_s=5.0)
        mastered, _, _ = master_audio(audio, SR)
        assert mastered.shape == audio.shape

    def test_output_dtype_float32(self):
        audio = _make_pink_noise(duration_s=5.0)
        mastered, _, _ = master_audio(audio, SR)
        assert mastered.dtype == np.float32

    def test_lufs_near_target(self):
        """After mastering, LUFS should be within tolerance of -14."""
        import pyloudnorm as pyln
        audio    = _make_pink_noise(duration_s=10.0, lufs_target=-20.0)
        mastered, _, _ = master_audio(audio, SR)
        meter    = pyln.Meter(SR)
        lufs_out = meter.integrated_loudness(mastered.T)
        assert abs(lufs_out - TARGET_LUFS) <= LUFS_TOLERANCE + 0.3

    def test_true_peak_within_ceiling(self):
        """True peak of the mastered output must not exceed the ceiling."""
        audio = _make_pink_noise(duration_s=5.0)
        mastered, _, tp = master_audio(audio, SR)
        assert tp <= TRUE_PEAK_CEIL + 0.2

    def test_silence_handled_gracefully(self):
        """Silent audio should not raise — LUFS will be -inf or very negative."""
        silence = np.zeros((2, SR * 5), dtype=np.float32)
        mastered, lufs, tp = master_audio(silence, SR)
        assert mastered.shape == silence.shape
        # Should not raise; LUFS can be -inf for silence

    def test_mono_input_converted_to_stereo(self):
        """1-D mono audio should be treated as stereo (duplicated channels)."""
        mono  = np.random.randn(SR * 5).astype(np.float32) * 0.1
        out, _, _ = master_audio(mono, SR)
        assert out.ndim == 2
        assert out.shape[0] == 2


class TestRunAudioQA:
    def _write_audio(self, audio: np.ndarray, tmp_path: Path) -> str:
        path = str(tmp_path / "test.wav")
        sf.write(path, audio.T, SR, subtype="PCM_24")
        return path

    def test_passes_on_correctly_mastered_audio(self, tmp_path: Path):
        audio   = _make_pink_noise(duration_s=10.0)
        mastered, _, _ = master_audio(audio, SR)
        path    = self._write_audio(mastered, tmp_path)
        result  = run_audio_qa("test_mix", path, target_lufs=TARGET_LUFS)
        assert result.passed, f"QA failed: {result.issues}"

    def test_fails_on_loud_audio(self, tmp_path: Path):
        # Create audio that's much louder than -14 LUFS
        loud = _make_pink_noise(duration_s=10.0, lufs_target=-5.0)
        path = self._write_audio(loud, tmp_path)
        result = run_audio_qa("test_mix", path, target_lufs=TARGET_LUFS)
        assert not result.passed
        assert any("LUFS" in issue for issue in result.issues)

    def test_fails_on_missing_file(self):
        result = run_audio_qa("test_mix", "/nonexistent/file.wav")
        assert not result.passed
        assert len(result.issues) > 0

    def test_returns_measured_values(self, tmp_path: Path):
        audio   = _make_pink_noise(duration_s=10.0)
        mastered, _, _ = master_audio(audio, SR)
        path    = self._write_audio(mastered, tmp_path)
        result  = run_audio_qa("test_mix", path)
        assert result.lufs_measured != 0.0
        assert result.true_peak_dbfs != 0.0
