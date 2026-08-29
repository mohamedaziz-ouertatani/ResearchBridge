"""add citation_fetch_runs (run history for rb-citations-fetch --all)

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-29
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "citation_fetch_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("papers_seen", sa.Integer(), server_default="0", nullable=False),
        sa.Column("papers_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("edges_created", sa.Integer(), server_default="0", nullable=False),
        sa.Column("edges_already_existed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("citation_fetch_runs")
