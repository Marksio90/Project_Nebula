"""Convert JSON→JSONB for platform_uploads and add created_at indexes

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-08 00:00:00.000000

Why:
  1. tags_pl / chapters_pl were stored as plain JSON (text internally in PG).
     JSONB stores the data in a binary decomposed format: faster reads,
     supports GIN indexing, and compresses better than text JSON.

  2. All TimestampMixin tables sort/filter by created_at DESC.
     Without an index, ORDER BY created_at forces a full sequential scan —
     this degrades linearly as the catalogue grows.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. JSON → JSONB for platform_uploads ──────────────────────────────
    op.alter_column(
        "platform_uploads",
        "tags_pl",
        type_=JSONB,
        postgresql_using="tags_pl::jsonb",
        existing_nullable=True,
    )
    op.alter_column(
        "platform_uploads",
        "chapters_pl",
        type_=JSONB,
        postgresql_using="chapters_pl::jsonb",
        existing_nullable=True,
    )

    # ── 2. created_at indexes on all TimestampMixin tables ─────────────────
    for table in ("mixes", "stems", "visuals", "viral_shorts", "platform_uploads"):
        op.create_index(
            f"ix_{table}_created_at",
            table,
            ["created_at"],
        )


def downgrade() -> None:
    for table in ("mixes", "stems", "visuals", "viral_shorts", "platform_uploads"):
        op.drop_index(f"ix_{table}_created_at", table_name=table)

    op.alter_column(
        "platform_uploads",
        "chapters_pl",
        type_=sa.JSON,
        postgresql_using="chapters_pl::text::json",
        existing_nullable=True,
    )
    op.alter_column(
        "platform_uploads",
        "tags_pl",
        type_=sa.JSON,
        postgresql_using="tags_pl::text::json",
        existing_nullable=True,
    )
