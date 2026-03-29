"""
shared/db/models.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy 2.x declarative models for Project Nebula.

Table architecture:
  mixes               — Top-level pipeline record, one per generated mix
  stems               — Individual 30-second audio stems (90-96 per mix)
  visuals             — Images / video loops generated for each mix
  viral_shorts        — RMS-selected 60-second clips sliced for Shorts/TikTok
  platform_uploads    — Upload records per platform per content piece
  bpm_subgenre_registry — Dedup registry: mathematically guarantees 0% repetition
─────────────────────────────────────────────────────────────────────────────
"""

import enum
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enumerations ──────────────────────────────────────────────────────────────

class MixStatus(str, enum.Enum):
    PENDING          = "pending"
    STRATEGISING     = "strategising"    # CSO agent running
    PROMPT_GEN       = "prompt_gen"      # Prompt Engineer agents running
    FETCHING_STEMS   = "fetching_stems"  # Gemini Lyria 3 calls in flight
    FETCHING_VISUALS = "fetching_visuals"
    STITCHING        = "stitching"       # DSP Worker: beat-match + crossfade
    MASTERING        = "mastering"       # DSP Worker: pedalboard chain
    QA_AUDIO         = "qa_audio"
    RENDERING        = "rendering"       # Video Worker: FFmpeg full mix
    SLICING          = "slicing"         # Video Worker: viral shorts
    QA_VIDEO         = "qa_video"
    UPLOADING        = "uploading"
    COMPLETE         = "complete"
    FAILED           = "failed"


class StemStatus(str, enum.Enum):
    PENDING    = "pending"
    GENERATING = "generating"
    READY      = "ready"
    FAILED     = "failed"


class VisualType(str, enum.Enum):
    BACKGROUND_IMAGE    = "background_image"     # 16:9 Nano Banana 2 image
    VIDEO_LOOP          = "video_loop"           # 16:9 Veo video loop
    THUMBNAIL           = "thumbnail"            # 16:9 YouTube thumbnail
    SHORT_BACKGROUND    = "short_background"     # 9:16 Shorts/TikTok background
    SHORT_THUMBNAIL     = "short_thumbnail"      # 9:16 Shorts/TikTok cover


class VisualStatus(str, enum.Enum):
    PENDING    = "pending"
    GENERATING = "generating"
    READY      = "ready"
    FAILED     = "failed"


class Platform(str, enum.Enum):
    YOUTUBE = "youtube"
    TIKTOK  = "tiktok"


class ContentType(str, enum.Enum):
    FULL_MIX = "full_mix"
    SHORT    = "short"


class UploadStatus(str, enum.Enum):
    PENDING   = "pending"
    UPLOADING = "uploading"
    UPLOADED  = "uploaded"
    FAILED    = "failed"


# ── Mixins ────────────────────────────────────────────────────────────────────

class TimestampMixin:
    """Adds created_at / updated_at columns with server-side defaults."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# ── Mix ───────────────────────────────────────────────────────────────────────

class Mix(TimestampMixin, Base):
    """
    Top-level record for a single autonomous mix generation run.
    One Mix → many Stems, Visuals, ViralShorts, PlatformUploads.
    """
    __tablename__ = "mixes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), index=True)
    status: Mapped[MixStatus] = mapped_column(
        Enum(MixStatus, name="mix_status"), default=MixStatus.PENDING, index=True
    )

    # ── CSO-determined parameters ──────────────────────────────────────────
    requested_duration_minutes: Mapped[int]    = mapped_column(Integer, default=45)
    actual_duration_seconds:    Mapped[float | None] = mapped_column(Float)
    style_hint:     Mapped[str | None] = mapped_column(Text)
    bpm:            Mapped[float | None] = mapped_column(Float)
    subgenre:       Mapped[str | None]  = mapped_column(String(128), index=True)
    key_signature:  Mapped[str | None]  = mapped_column(String(16))
    stem_count:     Mapped[int]         = mapped_column(Integer, default=0)

    # ── File artefacts ─────────────────────────────────────────────────────
    mastered_audio_path: Mapped[str | None] = mapped_column(Text)
    full_video_path:     Mapped[str | None] = mapped_column(Text)

    # ── QA measurements ────────────────────────────────────────────────────
    lufs_measured:       Mapped[float | None] = mapped_column(Float)
    true_peak_dbfs:      Mapped[float | None] = mapped_column(Float)
    qa_passed:           Mapped[bool]         = mapped_column(Boolean, default=False)

    # ── Error tracking ─────────────────────────────────────────────────────
    error_message:  Mapped[str | None] = mapped_column(Text)
    retry_count:    Mapped[int]        = mapped_column(Integer, default=0)

    # ── Relationships ──────────────────────────────────────────────────────
    stems:            Mapped[list["Stem"]]           = relationship(back_populates="mix", cascade="all, delete-orphan", order_by="Stem.position")
    visuals:          Mapped[list["Visual"]]         = relationship(back_populates="mix", cascade="all, delete-orphan")
    viral_shorts:     Mapped[list["ViralShort"]]     = relationship(back_populates="mix", cascade="all, delete-orphan", order_by="ViralShort.rms_db.desc()")
    platform_uploads: Mapped[list["PlatformUpload"]] = relationship(back_populates="mix", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Mix id={self.id} bpm={self.bpm} subgenre={self.subgenre} status={self.status}>"


# ── Stem ──────────────────────────────────────────────────────────────────────

class Stem(TimestampMixin, Base):
    """
    A single 30-second audio stem generated by Gemini Lyria 3.
    Ordered by `position` within a Mix. Beat-alignment data stored here
    so the DSP worker can rebuild the stitch without re-running librosa.
    """
    __tablename__ = "stems"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    mix_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-indexed order in mix

    # ── Generation data ────────────────────────────────────────────────────
    gemini_prompt:  Mapped[str]        = mapped_column(Text, nullable=False)
    file_path:      Mapped[str | None] = mapped_column(Text)
    status:         Mapped[StemStatus] = mapped_column(
        Enum(StemStatus, name="stem_status"), default=StemStatus.PENDING, index=True
    )
    error_message:  Mapped[str | None] = mapped_column(Text)

    # ── DSP analysis data (written by librosa in dsp_worker) ───────────────
    bpm_detected:         Mapped[float | None] = mapped_column(Float)
    beat_offset_samples:  Mapped[int | None]   = mapped_column(BigInteger)  # sample-accurate alignment
    duration_seconds:     Mapped[float | None] = mapped_column(Float)
    rms_db:               Mapped[float | None] = mapped_column(Float)
    sample_rate:          Mapped[int | None]   = mapped_column(Integer)     # e.g. 44100 or 48000

    # ── Relationships ──────────────────────────────────────────────────────
    mix: Mapped["Mix"] = relationship(back_populates="stems")

    __table_args__ = (
        UniqueConstraint("mix_id", "position", name="uq_stem_mix_position"),
        Index("ix_stems_mix_status", "mix_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<Stem pos={self.position} bpm={self.bpm_detected} status={self.status}>"


# ── Visual ────────────────────────────────────────────────────────────────────

class Visual(TimestampMixin, Base):
    """
    A single visual asset (image or video loop) generated by Gemini
    (Nano Banana 2 for images, Veo for video loops).
    """
    __tablename__ = "visuals"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    mix_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    visual_type: Mapped[VisualType] = mapped_column(
        Enum(VisualType, name="visual_type"), nullable=False
    )
    aspect_ratio: Mapped[str] = mapped_column(String(8), default="16:9")  # "16:9" | "9:16"

    gemini_prompt: Mapped[str]        = mapped_column(Text, nullable=False)
    file_path:     Mapped[str | None] = mapped_column(Text)
    status:        Mapped[VisualStatus] = mapped_column(
        Enum(VisualStatus, name="visual_status"), default=VisualStatus.PENDING
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    mix: Mapped["Mix"] = relationship(back_populates="visuals")

    def __repr__(self) -> str:
        return f"<Visual type={self.visual_type} ratio={self.aspect_ratio} status={self.status}>"


# ── Viral Short ───────────────────────────────────────────────────────────────

class ViralShort(TimestampMixin, Base):
    """
    A 60-second 9:16 clip sliced from the highest-RMS segments of the mix.
    The Content Slicer agent creates up to 3 per mix (top RMS windows).
    """
    __tablename__ = "viral_shorts"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    mix_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank:              Mapped[int]         = mapped_column(Integer)           # 1=loudest drop
    start_seconds:     Mapped[float]       = mapped_column(Float, nullable=False)
    duration_seconds:  Mapped[float]       = mapped_column(Float, default=60.0)
    rms_db:            Mapped[float]       = mapped_column(Float, nullable=False)
    video_path:        Mapped[str | None]  = mapped_column(Text)
    upload_status:     Mapped[UploadStatus] = mapped_column(
        Enum(UploadStatus, name="short_upload_status"), default=UploadStatus.PENDING
    )
    error_message:     Mapped[str | None]  = mapped_column(Text)

    mix:              Mapped["Mix"]                   = relationship(back_populates="viral_shorts")
    platform_uploads: Mapped[list["PlatformUpload"]]  = relationship(back_populates="viral_short", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ViralShort rank={self.rank} start={self.start_seconds:.1f}s rms={self.rms_db:.1f}dB>"


# ── Platform Upload ───────────────────────────────────────────────────────────

class PlatformUpload(TimestampMixin, Base):
    """
    Tracks a single upload attempt to YouTube or TikTok.
    Polish metadata (title, description, tags, chapters) stored here.
    """
    __tablename__ = "platform_uploads"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    mix_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    viral_short_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("viral_shorts.id", ondelete="SET NULL"), nullable=True
    )
    platform:      Mapped[Platform]     = mapped_column(Enum(Platform,     name="platform"),      nullable=False)
    content_type:  Mapped[ContentType]  = mapped_column(Enum(ContentType,  name="content_type"),  nullable=False)
    upload_status: Mapped[UploadStatus] = mapped_column(Enum(UploadStatus, name="upload_status"), default=UploadStatus.PENDING, index=True)

    # ── Polish SEO metadata (FRONTEND LOCALIZATION) ────────────────────────
    title_pl:       Mapped[str | None]  = mapped_column(Text)   # Viral-optimised Polish title
    description_pl: Mapped[str | None]  = mapped_column(Text)   # Polish SEO description
    tags_pl:        Mapped[Any | None]  = mapped_column(JSON)   # list[str] — Polish hashtags
    chapters_pl:    Mapped[Any | None]  = mapped_column(JSON)   # list[{"time": "00:00", "title": "..."}]

    platform_video_id:    Mapped[str | None]      = mapped_column(String(255))
    platform_video_url:   Mapped[str | None]      = mapped_column(Text)
    scheduled_publish_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at:         Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message:        Mapped[str | None]      = mapped_column(Text)

    mix:         Mapped["Mix"]                  = relationship(back_populates="platform_uploads")
    viral_short: Mapped["ViralShort | None"]    = relationship(back_populates="platform_uploads")

    def __repr__(self) -> str:
        return f"<PlatformUpload platform={self.platform} type={self.content_type} status={self.upload_status}>"


# ── BPM / Subgenre Dedup Registry ─────────────────────────────────────────────

class BpmSubgenreRegistry(Base):
    """
    Deduplication registry queried by the CSO agent before every mix.
    Stores every (bpm, subgenre, key_signature) combination that has been
    used, guaranteeing 0% content repetition across the label's catalogue.

    The CSO selects combinations NOT present in this table.
    """
    __tablename__ = "bpm_subgenre_registry"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    mix_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    bpm:            Mapped[float] = mapped_column(Float,       nullable=False)
    subgenre:       Mapped[str]   = mapped_column(String(128), nullable=False)
    key_signature:  Mapped[str]   = mapped_column(String(16),  nullable=False)
    used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        # Composite uniqueness: same BPM + subgenre + key will never repeat
        UniqueConstraint("bpm", "subgenre", "key_signature", name="uq_bpm_subgenre_key"),
        # Fast CSO exclusion query: WHERE subgenre = ? AND bpm BETWEEN ? AND ?
        Index("ix_registry_subgenre_bpm", "subgenre", "bpm"),
    )

    def __repr__(self) -> str:
        return f"<BpmSubgenreRegistry bpm={self.bpm} subgenre={self.subgenre} key={self.key_signature}>"
