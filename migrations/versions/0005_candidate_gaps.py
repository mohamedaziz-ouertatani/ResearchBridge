"""candidate_gaps, candidate_gap_evidence (Phase 3 explicit->implicit gaps)

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candidate_gaps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("seed_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("gap_type", sa.String(), nullable=False, server_default="inference"),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("contributing_paper_count", sa.Integer(), nullable=False),
        sa.Column("similarity_threshold", sa.Float(), nullable=False),
        sa.Column("detection_method", sa.String(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_candidate_gaps_seed_paper_id", "candidate_gaps", ["seed_paper_id"])

    op.create_table(
        "candidate_gap_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "candidate_gap_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("candidate_gaps.id"), nullable=False
        ),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id"), nullable=False),
        sa.UniqueConstraint("candidate_gap_id", "evidence_id", name="uq_candidate_gap_evidence_gap_evidence"),
    )
    op.create_index("ix_candidate_gap_evidence_gap_id", "candidate_gap_evidence", ["candidate_gap_id"])


def downgrade() -> None:
    op.drop_table("candidate_gap_evidence")
    op.drop_table("candidate_gaps")
