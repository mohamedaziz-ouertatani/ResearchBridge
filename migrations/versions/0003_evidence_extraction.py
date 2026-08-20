"""evidence, extracted_claims, extraction_runs, extraction_errors

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("evidence_type", sa.String(), nullable=False),
        sa.Column("section", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source_locator", sa.String(), nullable=True),
        sa.Column("extraction_method", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_evidence_paper_id", "evidence", ["paper_id"])

    op.create_table(
        "extracted_claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("claim_type", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id"), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_extracted_claims_paper_id", "extracted_claims", ["paper_id"])

    op.create_table(
        "extraction_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("extractor_name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("papers_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("claims_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("candidates_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
    )

    op.create_table(
        "extraction_errors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "extraction_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("extraction_runs.id"),
            nullable=False,
        ),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("error_type", sa.String(), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_extraction_errors_run_id", "extraction_errors", ["extraction_run_id"])


def downgrade() -> None:
    op.drop_table("extraction_errors")
    op.drop_table("extraction_runs")
    op.drop_table("extracted_claims")
    op.drop_table("evidence")
