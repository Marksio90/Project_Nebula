"""Add transition_arc column to mixes table

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-04 00:00:00.000000

Why: The CSO agent generates a transition_arc describing the musical narrative
(intro timestamps, peak windows, outro) as part of CSOStrategy.  Storing it on
the Mix row lets the video renderer generate chapter title overlays at render
time (step 8) without waiting for the Polish SEO agent (step 11), which is the
correct place for SEO chapter markers (YouTube description) but runs too late
to burn text into the video.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("mixes", sa.Column("transition_arc", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("mixes", "transition_arc")
