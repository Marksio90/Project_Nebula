"""Initial schema — all Project Nebula tables

Revision ID: 0001
Revises:
Create Date: 2026-03-29 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Custom ENUM types ─────────────────────────────────────────────────────
    mix_status = sa.Enum(
        "pending", "strategising", "prompt_gen", "fetching_stems", "fetching_visuals",
        "stitching", "mastering", "qa_audio", "rendering", "slicing", "qa_video",
        "uploading", "complete", "failed",
        name="mix_status",
    )
    stem_status    = sa.Enum("pending", "generating", "ready", "failed", name="stem_status")
    visual_type    = sa.Enum(
        "background_image", "video_loop", "thumbnail", "short_background", "short_thumbnail",
        name="visual_type",
    )
    visual_status  = sa.Enum("pending", "generating", "ready", "failed", name="visual_status")
    platform_enum  = sa.Enum("youtube", "tiktok", name="platform")
    content_type   = sa.Enum("full_mix", "short", name="content_type")
    upload_status  = sa.Enum("pending", "uploading", "uploaded", "failed", name="upload_status")
    short_upload_status = sa.Enum("pending", "uploading", "uploaded", "failed", name="short_upload_status")

    for e in (mix_status, stem_status, visual_type, visual_status,
              platform_enum, content_type, upload_status, short_upload_status):
        e.create(op.get_bind(), checkfirst=True)

    # ── mixes ─────────────────────────────────────────────────────────────────
    op.create_table(
        "mixes",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("status", sa.Enum(name="mix_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("requested_duration_minutes", sa.Integer, nullable=False, server_default="45"),
        sa.Column("actual_duration_seconds", sa.Float, nullable=True),
        sa.Column("style_hint", sa.Text, nullable=True),
        sa.Column("bpm", sa.Float, nullable=True),
        sa.Column("subgenre", sa.String(128), nullable=True),
        sa.Column("key_signature", sa.String(16), nullable=True),
        sa.Column("stem_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("mastered_audio_path", sa.Text, nullable=True),
        sa.Column("full_video_path", sa.Text, nullable=True),
        sa.Column("lufs_measured", sa.Float, nullable=True),
        sa.Column("true_peak_dbfs", sa.Float, nullable=True),
        sa.Column("qa_passed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_mixes_status",   "mixes", ["status"])
    op.create_index("ix_mixes_subgenre", "mixes", ["subgenre"])
    op.create_index("ix_mixes_celery",   "mixes", ["celery_task_id"])

    # ── stems ─────────────────────────────────────────────────────────────────
    op.create_table(
        "stems",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("mix_id", UUID(as_uuid=False), sa.ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("gemini_prompt", sa.Text, nullable=False),
        sa.Column("file_path", sa.Text, nullable=True),
        sa.Column("status", sa.Enum(name="stem_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("bpm_detected", sa.Float, nullable=True),
        sa.Column("beat_offset_samples", sa.BigInteger, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("rms_db", sa.Float, nullable=True),
        sa.Column("sample_rate", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_stems_mix_id", "stems", ["mix_id"])
    op.create_index("ix_stems_status",  "stems", ["status"])
    op.create_index("ix_stems_mix_status", "stems", ["mix_id", "status"])
    op.create_unique_constraint("uq_stem_mix_position", "stems", ["mix_id", "position"])

    # ── visuals ───────────────────────────────────────────────────────────────
    op.create_table(
        "visuals",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("mix_id", UUID(as_uuid=False), sa.ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("visual_type", sa.Enum(name="visual_type", create_type=False), nullable=False),
        sa.Column("aspect_ratio", sa.String(8), nullable=False, server_default="16:9"),
        sa.Column("gemini_prompt", sa.Text, nullable=False),
        sa.Column("file_path", sa.Text, nullable=True),
        sa.Column("status", sa.Enum(name="visual_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_visuals_mix_id", "visuals", ["mix_id"])

    # ── viral_shorts ──────────────────────────────────────────────────────────
    op.create_table(
        "viral_shorts",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("mix_id", UUID(as_uuid=False), sa.ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("start_seconds", sa.Float, nullable=False),
        sa.Column("duration_seconds", sa.Float, nullable=False, server_default="60.0"),
        sa.Column("rms_db", sa.Float, nullable=False),
        sa.Column("video_path", sa.Text, nullable=True),
        sa.Column("upload_status", sa.Enum(name="short_upload_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_viral_shorts_mix_id", "viral_shorts", ["mix_id"])

    # ── platform_uploads ──────────────────────────────────────────────────────
    op.create_table(
        "platform_uploads",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("mix_id", UUID(as_uuid=False), sa.ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("viral_short_id", UUID(as_uuid=False), sa.ForeignKey("viral_shorts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("platform", sa.Enum(name="platform", create_type=False), nullable=False),
        sa.Column("content_type", sa.Enum(name="content_type", create_type=False), nullable=False),
        sa.Column("upload_status", sa.Enum(name="upload_status", create_type=False), nullable=False, server_default="pending"),
        sa.Column("title_pl", sa.Text, nullable=True),
        sa.Column("description_pl", sa.Text, nullable=True),
        sa.Column("tags_pl", sa.JSON, nullable=True),
        sa.Column("chapters_pl", sa.JSON, nullable=True),
        sa.Column("platform_video_id", sa.String(255), nullable=True),
        sa.Column("platform_video_url", sa.Text, nullable=True),
        sa.Column("scheduled_publish_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_platform_uploads_mix_id",       "platform_uploads", ["mix_id"])
    op.create_index("ix_platform_uploads_upload_status", "platform_uploads", ["upload_status"])

    # ── bpm_subgenre_registry ─────────────────────────────────────────────────
    op.create_table(
        "bpm_subgenre_registry",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column("mix_id", UUID(as_uuid=False), sa.ForeignKey("mixes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("bpm", sa.Float, nullable=False),
        sa.Column("subgenre", sa.String(128), nullable=False),
        sa.Column("key_signature", sa.String(16), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_registry_mix_id",       "bpm_subgenre_registry", ["mix_id"])
    op.create_index("ix_registry_subgenre_bpm", "bpm_subgenre_registry", ["subgenre", "bpm"])
    op.create_unique_constraint(
        "uq_bpm_subgenre_key", "bpm_subgenre_registry", ["bpm", "subgenre", "key_signature"]
    )


def downgrade() -> None:
    op.drop_table("bpm_subgenre_registry")
    op.drop_table("platform_uploads")
    op.drop_table("viral_shorts")
    op.drop_table("visuals")
    op.drop_table("stems")
    op.drop_table("mixes")

    for name in ("short_upload_status", "upload_status", "content_type", "platform",
                 "visual_status", "visual_type", "stem_status", "mix_status"):
        sa.Enum(name=name).drop(op.get_bind(), checkfirst=True)
