"""research_inputs, research_assessments, research_assessment_evidence
(ResearchInput -> ResearchAssessment vertical slice, blueprint Sec 2A)

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "research_inputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("input_type", sa.String(), nullable=False, server_default="idea"),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("matched_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=True),
        sa.Column("source_filename", sa.String(), nullable=True),
        sa.Column("submitted_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "research_assessments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "research_input_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_inputs.id"), nullable=False
        ),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("retrieved_paper_ids", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("comparison_summary", sa.Text(), nullable=True),
        sa.Column("novelty_level", sa.String(), nullable=False, server_default="not_assessed"),
        sa.Column("novelty_reasoning", sa.Text(), nullable=True),
        sa.Column("research_gap_text", sa.Text(), nullable=True),
        sa.Column("research_gap_source", sa.String(), nullable=True),
        sa.Column("candidate_gap_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("candidate_gaps.id"), nullable=True),
        sa.Column("potential_applications", postgresql.JSONB(), nullable=True),
        sa.Column("potential_opportunities", postgresql.JSONB(), nullable=True),
        sa.Column("technical_feasibility_level", sa.String(), nullable=False, server_default="not_assessed"),
        sa.Column("risks_and_limitations", sa.Text(), nullable=True),
        sa.Column("external_validation_needed", sa.Text(), nullable=True),
        sa.Column("recommendation", sa.String(), nullable=True),
        sa.Column("confidence", sa.String(), nullable=True),
        sa.Column("human_reviewed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("ix_research_assessments_research_input_id", "research_assessments", ["research_input_id"])

    op.create_table(
        "research_assessment_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "research_assessment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_assessments.id"),
            nullable=False,
        ),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "research_assessment_id", "evidence_id", "role", name="uq_research_assessment_evidence_assessment_evidence_role"
        ),
    )
    op.create_index(
        "ix_research_assessment_evidence_assessment_id", "research_assessment_evidence", ["research_assessment_id"]
    )


def downgrade() -> None:
    op.drop_table("research_assessment_evidence")
    op.drop_table("research_assessments")
    op.drop_table("research_inputs")
