"""
shared/schemas/events.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 schemas that flow between Celery tasks as JSON payloads.
These are the "contracts" between pipeline stages — strongly typed so any
mismatch is caught immediately rather than deep inside a task.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ── Shared helpers ────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = {"frozen": True}  # Immutable — tasks must not mutate events


# ── Pipeline trigger ──────────────────────────────────────────────────────────

class MixPipelineRequest(_Base):
    """Payload that kicks off the entire orchestrate_mix_pipeline task."""
    mix_id:                     str
    genre:                      str                  # User-selected genre from GENRE_NAMES
    requested_duration_minutes: int = Field(ge=10, le=300)  # Set autonomously by orchestrator


# ── CSO strategy output ───────────────────────────────────────────────────────

class CSOStrategy(_Base):
    """
    Result of the Chief Strategy Officer agent.
    Drives every downstream agent in the pipeline.
    """
    mix_id:                     str
    bpm:                        float     = Field(ge=60.0, le=220.0)
    subgenre:                   str       = Field(min_length=2, max_length=128)
    key_signature:              str       = Field(min_length=1, max_length=16)  # e.g. "D minor"
    style_description:          str       # English — feeds Prompt Engineers
    transition_arc:             str       # English — e.g. "Dark Neurofunk → Liquid DnB at ~22 min"
    stem_count:                 int       = Field(ge=14, le=150)
    requested_duration_minutes: int

    @field_validator("bpm")
    @classmethod
    def round_bpm(cls, v: float) -> float:
        return round(v, 1)


# ── Audio prompt batch ────────────────────────────────────────────────────────

class StemPrompt(_Base):
    """A single Gemini Lyria 3 prompt for one 30-second stem."""
    position:       int           # 0-indexed position in the mix
    prompt_en:      str           # English prompt → Gemini Lyria 3
    transition_type: str          # "intro" | "build" | "drop" | "breakdown" | "outro"
    intensity:      float         = Field(ge=0.0, le=1.0)  # 0=chill, 1=peak


class AudioPromptBatch(_Base):
    mix_id:    str
    strategy:  CSOStrategy
    prompts:   list[StemPrompt]


# ── Visual prompt batch ───────────────────────────────────────────────────────

class VisualPrompt(_Base):
    visual_type:   str   # maps to VisualType enum values
    aspect_ratio:  str   # "16:9" | "9:16"
    prompt_en:     str   # English prompt → Gemini Nano Banana 2 / Veo


class VisualPromptBatch(_Base):
    mix_id:   str
    prompts:  list[VisualPrompt]


# ── Stem fetch result ─────────────────────────────────────────────────────────

class StemFetchResult(_Base):
    mix_id:    str
    stem_id:   str
    position:  int
    file_path: str
    status:    str    # "ready" | "failed"
    error:     str | None = None


class StemBatchResult(_Base):
    mix_id:  str
    results: list[StemFetchResult]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == "ready")

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.success_count


# ── DSP stitch result ─────────────────────────────────────────────────────────

class AudioStitchResult(_Base):
    mix_id:               str
    mastered_audio_path:  str
    actual_duration_seconds: float
    lufs_integrated:      float   # measured after mastering
    true_peak_dbfs:       float
    stem_count_used:      int


# ── QA results ────────────────────────────────────────────────────────────────

class AudioQAResult(_Base):
    mix_id:          str
    passed:          bool
    lufs_measured:   float
    true_peak_dbfs:  float
    issues:          list[str] = Field(default_factory=list)


class VideoQAResult(_Base):
    mix_id:          str
    passed:          bool
    frame_drop_rate: float   # fraction 0.0–1.0
    issues:          list[str] = Field(default_factory=list)


# ── Video render result ───────────────────────────────────────────────────────

class VideoRenderResult(_Base):
    mix_id:           str
    full_video_path:  str
    duration_seconds: float
    resolution:       str   # e.g. "3840x2160"
    codec:            str   # e.g. "h264_nvenc" | "libx264"


# ── Viral short slice result ──────────────────────────────────────────────────

class ViralShortResult(_Base):
    mix_id:          str
    short_id:        str
    rank:            int
    start_seconds:   float
    rms_db:          float
    video_path:      str


class ViralSliceResult(_Base):
    mix_id:  str
    shorts:  list[ViralShortResult]


# ── Polish SEO metadata (FRONTEND LOCALIZATION) ───────────────────────────────

class ChapterMarker(_Base):
    time_str: str    # e.g. "00:00", "05:30", "22:15"
    title_pl: str    # Polish chapter title


class PolishSEOMetadata(_Base):
    """
    All user-facing content for YouTube / TikTok.
    MUST be in viral-optimised Polish (język polski).
    """
    mix_id:            str
    title_pl:          str                   # Max 100 chars — YouTube title
    description_pl:    str                   # Max 5000 chars — SEO description
    tags_pl:           list[str]             # Max 500 total chars — hashtags
    chapters_pl:       list[ChapterMarker]   # Timestamp chapter markers
    shorts_titles_pl:  list[str]             # 3 titles, one per viral short


# ── Upload result ─────────────────────────────────────────────────────────────

class UploadResult(_Base):
    mix_id:             str
    upload_id:          str
    platform:           str
    content_type:       str
    platform_video_id:  str | None = None
    platform_video_url: str | None = None
    status:             str
    error:              str | None = None
