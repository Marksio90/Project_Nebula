"""
tests/unit/test_tasks/test_pipeline_events.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for Pydantic event schemas — validates payload contracts between
pipeline stages without requiring a live DB or Celery broker.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from pydantic import ValidationError

from shared.schemas.events import (
    AudioPromptBatch,
    CSOStrategy,
    MixPipelineRequest,
    PolishSEOMetadata,
    StemPrompt,
    ViralShortResult,
    ViralSliceResult,
)


class TestMixPipelineRequest:
    def test_valid_defaults(self):
        req = MixPipelineRequest(mix_id="abc-123")
        assert req.requested_duration_minutes == 45
        assert req.style_hint is None
        assert req.force_bpm is None

    def test_duration_bounds(self):
        with pytest.raises(ValidationError):
            MixPipelineRequest(mix_id="x", requested_duration_minutes=5)  # < 10
        with pytest.raises(ValidationError):
            MixPipelineRequest(mix_id="x", requested_duration_minutes=150)  # > 120

    def test_force_bpm_bounds(self):
        req = MixPipelineRequest(mix_id="x", force_bpm=174)
        assert req.force_bpm == 174
        with pytest.raises(ValidationError):
            MixPipelineRequest(mix_id="x", force_bpm=300)  # > 220


class TestCSOStrategy:
    def test_bpm_rounded(self):
        s = CSOStrategy(
            mix_id="x",
            bpm=174.056789,
            subgenre="Neurofunk",
            key_signature="D minor",
            style_description="Dark and heavy",
            transition_arc="Neurofunk to Liquid at 22min",
            stem_count=90,
            requested_duration_minutes=45,
        )
        assert s.bpm == 174.1

    def test_stem_count_bounds(self):
        with pytest.raises(ValidationError):
            CSOStrategy(
                mix_id="x", bpm=174, subgenre="x", key_signature="C",
                style_description="x", transition_arc="x",
                stem_count=200,  # > 96
                requested_duration_minutes=45,
            )


class TestAudioPromptBatch:
    def test_ordered_positions(self):
        strategy = CSOStrategy(
            mix_id="m1", bpm=174.0, subgenre="Neurofunk",
            key_signature="D minor", style_description="Dark",
            transition_arc="Neurofunk → Liquid", stem_count=90,
            requested_duration_minutes=45,
        )
        batch = AudioPromptBatch(
            mix_id="m1",
            strategy=strategy,
            prompts=[
                StemPrompt(position=0, prompt_en="intro", transition_type="intro", intensity=0.2),
                StemPrompt(position=1, prompt_en="drop",  transition_type="drop",  intensity=0.9),
                StemPrompt(position=2, prompt_en="outro", transition_type="outro", intensity=0.3),
            ],
        )
        assert len(batch.prompts) == 3
        assert batch.prompts[1].intensity == 0.9


class TestPolishSEOMetadata:
    def test_requires_mix_id(self):
        with pytest.raises(ValidationError):
            PolishSEOMetadata(
                # missing mix_id
                title_pl="Test",
                description_pl="Opis",
                tags_pl=["#dnb"],
                chapters_pl=[],
                shorts_titles_pl=["a", "b", "c"],
            )

    def test_valid_metadata(self):
        seo = PolishSEOMetadata(
            mix_id="m1",
            title_pl="🔥 2 GODZINY Neurofunk DnB | Mix 2026",
            description_pl="Najlepsza muzyka drum and bass...",
            tags_pl=["#drumandbass", "#neurofunk", "#mix2026"],
            chapters_pl=[],
            shorts_titles_pl=["Drop 1", "Drop 2", "Drop 3"],
        )
        assert "Neurofunk" in seo.title_pl


class TestViralSliceResult:
    def test_three_shorts(self):
        result = ViralSliceResult(
            mix_id="m1",
            shorts=[
                ViralShortResult(mix_id="m1", short_id="s1", rank=1, start_seconds=120.0, rms_db=-6.0, video_path="/short1.mp4"),
                ViralShortResult(mix_id="m1", short_id="s2", rank=2, start_seconds=900.0, rms_db=-7.5, video_path="/short2.mp4"),
                ViralShortResult(mix_id="m1", short_id="s3", rank=3, start_seconds=1800.0, rms_db=-8.1, video_path="/short3.mp4"),
            ],
        )
        assert len(result.shorts) == 3
        assert result.shorts[0].rank == 1
