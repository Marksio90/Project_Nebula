"""
services/dsp_worker/audio/stem_stitcher.py
─────────────────────────────────────────────────────────────────────────────
THE CORE DSP ENGINE — Beat-matched stem stitching with on-the-one crossfades.

Pipeline (called by the `stitch_and_master_audio` Celery task):
  1. Load all READY stems for the mix from PostgreSQL (ordered by position).
  2. For each stem: run load_and_analyse_stem() → BeatAlignedStem.
  3. Update each Stem row with librosa measurements (bpm_detected,
     beat_offset_samples, rms_db, sample_rate).
  4. Concatenate stems with 1-beat raised-cosine crossfades on the downbeat.
  5. Run the full pedalboard mastering chain + LUFS normalisation.
  6. Export as both WAV (archive) and MP3 (streaming) to MIXES_DIR.
  7. Return AudioStitchResult for the QA task.

Crossfade algorithm (visual):

  Stem A: [...body...| fade_out →]
  Stem B:             [← fade_in |...body...]
  Mixed:  [...body...| blend zone |...body...]

  The blend zone is exactly 1 beat period centred on the join downbeat.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy import select, update

from services.dsp_worker.audio.beat_aligner import (
    BeatAlignedStem,
    compute_crossfade_join,
    load_and_analyse_stem,
)
from services.dsp_worker.audio.mastering import master_audio
from shared.config import get_settings
from shared.db.models import Mix, Stem, StemStatus
from shared.db.session import get_sync_db
from shared.schemas.events import AudioStitchResult

log      = logging.getLogger("nebula.dsp.stitcher")
settings = get_settings()

WAV_SUBTYPE  = "PCM_24"   # 24-bit WAV for the archive master
MP3_BITRATE  = "320k"     # Exported via soundfile + external encoder


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def stitch_and_master(mix_id: str) -> AudioStitchResult:
    """
    Full beat-match + master pipeline for a single mix.
    Called directly from the `stitch_and_master_audio` Celery task.
    """
    log.info("▶ stitch_and_master: mix_id=%s", mix_id)

    # ── 1. Load stem records from DB ──────────────────────────────────────
    with get_sync_db() as db:
        mix = db.execute(select(Mix).where(Mix.id == mix_id)).scalar_one()
        stem_rows = db.execute(
            select(Stem)
            .where(Stem.mix_id == mix_id, Stem.status == StemStatus.READY)
            .order_by(Stem.position)
        ).scalars().all()

    if not stem_rows:
        raise RuntimeError(f"No READY stems found for mix_id={mix_id}")

    target_bpm = float(mix.bpm or 174.0)
    log.info("Stitching %d stems at %.1f BPM", len(stem_rows), target_bpm)

    # ── 2. Beat-analyse all stems ─────────────────────────────────────────
    aligned_stems: list[BeatAlignedStem] = []
    for row in stem_rows:
        if not row.file_path or not Path(row.file_path).exists():
            log.warning("Stem %04d: file missing at %s — skipping", row.position, row.file_path)
            continue
        try:
            stem = load_and_analyse_stem(
                file_path=row.file_path,
                target_bpm=target_bpm,
                position=row.position,
            )
            aligned_stems.append(stem)

            # Persist DSP measurements back to DB
            with get_sync_db() as db:
                db.execute(
                    update(Stem).where(Stem.id == row.id).values(
                        bpm_detected=stem.bpm_detected,
                        beat_offset_samples=int(stem.trim_offset_samples),
                        rms_db=stem.rms_db,
                        sample_rate=stem.sr,
                    )
                )
        except Exception as exc:
            log.error("Failed to analyse stem %04d: %s", row.position, exc)
            # Non-fatal: continue with remaining stems

    if len(aligned_stems) < 2:
        raise RuntimeError(
            f"Only {len(aligned_stems)} stems loaded — insufficient for a mix."
        )

    log.info("Beat analysis complete: %d/%d stems ready", len(aligned_stems), len(stem_rows))

    # ── 3. Concatenate with on-the-one crossfades ─────────────────────────
    stitched = _concatenate_with_crossfades(aligned_stems)
    sr       = aligned_stems[0].sr
    duration = stitched.shape[1] / sr
    log.info("Stitched duration: %.1f seconds (%.1f minutes)", duration, duration / 60)

    # ── 4. Mastering chain ────────────────────────────────────────────────
    log.info("Running mastering chain...")
    mastered, lufs_pre, true_peak = master_audio(
        audio=stitched,
        sr=sr,
        target_lufs=-14.0,
        true_peak_ceil=-1.0,
    )

    # ── 5. Export to disk ─────────────────────────────────────────────────
    out_dir = Path(settings.mixes_dir) / mix_id
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(out_dir / "master.wav")
    mp3_path = str(out_dir / "master.mp3")

    # Write 24-bit WAV archive
    sf.write(
        wav_path,
        mastered.T,     # soundfile expects (N, channels)
        sr,
        subtype=WAV_SUBTYPE,
    )
    log.info("WAV master written: %s", wav_path)

    # Write MP3 via soundfile (requires libsndfile with MP3 support)
    # Fallback: write second WAV if MP3 not supported
    try:
        sf.write(mp3_path, mastered.T, sr, format="MP3")
        log.info("MP3 master written: %s", mp3_path)
    except Exception as exc:
        log.warning("MP3 export failed (%s) — using WAV path as mastered_audio_path", exc)
        mp3_path = wav_path

    return AudioStitchResult(
        mix_id=mix_id,
        mastered_audio_path=wav_path,   # WAV is the canonical master
        actual_duration_seconds=round(duration, 2),
        lufs_integrated=round(lufs_pre, 3),
        true_peak_dbfs=round(true_peak, 3),
        stem_count_used=len(aligned_stems),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stitching engine
# ─────────────────────────────────────────────────────────────────────────────

def _concatenate_with_crossfades(stems: list[BeatAlignedStem]) -> np.ndarray:
    """
    Stitch N stems together using 1-beat crossfades aligned to the downbeat.

    Strategy per join (stem[i] → stem[i+1]):
      a. Take the body of stem[i]    (all samples except the last beat)
      b. Compute the blended overlap (1 beat: end of stem[i] + start of stem[i+1])
      c. After the first stem: prepend the body, then append the blend
         For subsequent stems: append body (minus first beat) then blend

    The resulting waveform:
      [stem0_body][blend01][stem1_body][blend12]...[stemN_body]

    All arithmetic is in samples; shapes are (2, N) float32.
    """
    if len(stems) == 1:
        return stems[0].audio.astype(np.float32)

    chunks: list[np.ndarray] = []

    for i, stem in enumerate(stems):
        fade_len = stem.beat_period_samples   # 1 beat = crossfade window

        if i == 0:
            # First stem: output body (everything except the last `fade_len` samples)
            body = stem.audio[:, :-fade_len]
            chunks.append(body.astype(np.float32))
        else:
            # For stems 2..N: body is everything BETWEEN the first beat
            # (which was consumed in the previous blend) and the last beat
            body_start = fade_len
            body_end   = stem.audio.shape[1] - fade_len if i < len(stems) - 1 else stem.audio.shape[1]
            body_end   = max(body_start, body_end)
            body       = stem.audio[:, body_start:body_end]
            chunks.append(body.astype(np.float32))

        # Compute the crossfade blend between stem[i] and stem[i+1]
        if i < len(stems) - 1:
            blend = compute_crossfade_join(
                stem_a=stem,
                stem_b=stems[i + 1],
                crossfade_beats=1,
            )
            chunks.append(blend.astype(np.float32))

    # Add the body of the last stem (no outgoing crossfade)
    # Already added in the loop above (when i == len-1, body_end = full length)

    stitched = np.concatenate(chunks, axis=1)
    log.debug(
        "Stitched: %d chunks → shape %s (%.1f min)",
        len(chunks), stitched.shape, stitched.shape[1] / (44100 * 60),
    )
    return stitched
