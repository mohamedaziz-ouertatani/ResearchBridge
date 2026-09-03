"""add gap_detection_runs (run history for rb-gaps-detect --all)

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-03
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gap_detection_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("papers_seen", sa.Integer(), server_default="0", nullable=False),
        sa.Column("papers_skipped", sa.Integer(), server_default="0", nullable=False),
        sa.Column("papers_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("no_relevant_papers", sa.Integer(), server_default="0", nullable=False),
        sa.Column("insufficient_evidence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("gaps_found", sa.Integer(), server_default="0", nullable=False),
        sa.Column("gaps_saved", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_table("gap_detection_runs")
