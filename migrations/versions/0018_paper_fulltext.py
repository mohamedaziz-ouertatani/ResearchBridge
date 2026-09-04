"""paper_fulltext, fulltext_fetch_runs, fulltext_fetch_errors (Sec 46)

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_fulltext",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("sections", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("fetch_method", sa.String(), nullable=False, server_default="pymupdf"),
        sa.Column("fetched_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("paper_id", name="uq_paper_fulltext_paper_id"),
    )

    op.create_table(
        "fulltext_fetch_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("papers_seen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_skipped_no_url", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.create_table(
        "fulltext_fetch_errors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "fulltext_fetch_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fulltext_fetch_runs.id"),
            nullable=False,
        ),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("error_type", sa.String(), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_fulltext_fetch_errors_run_id", "fulltext_fetch_errors", ["fulltext_fetch_run_id"])


def downgrade() -> None:
    op.drop_table("fulltext_fetch_errors")
    op.drop_table("fulltext_fetch_runs")
    op.drop_table("paper_fulltext")
