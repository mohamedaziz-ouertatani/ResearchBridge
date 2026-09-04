"""analysis_claims, claim_evidence (Sec 16): structured system-reasoning layer

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CLAIM_TYPE_CONSTRAINT = "ck_analysis_claims_claim_type"
STATUS_CONSTRAINT = "ck_analysis_claims_status"
RELATIONSHIP_CONSTRAINT = "ck_claim_evidence_relationship"


def upgrade() -> None:
    op.create_table(
        "analysis_claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("claim_type", sa.String(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("source_table", sa.String(), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_analysis_claims_source", "analysis_claims", ["source_table", "source_id"])
    op.create_check_constraint(
        CLAIM_TYPE_CONSTRAINT,
        "analysis_claims",
        "claim_type IN ('fact', 'inference', 'hypothesis', 'opportunity', 'speculation')",
    )
    op.create_check_constraint(
        STATUS_CONSTRAINT,
        "analysis_claims",
        "status IN ('pending', 'approved', 'rejected')",
    )

    op.create_table(
        "claim_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("analysis_claims.id"), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("evidence.id"), nullable=False),
        sa.Column("relationship", sa.String(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("claim_id", "evidence_id", "relationship", name="uq_claim_evidence_claim_evidence_relationship"),
    )
    op.create_index("ix_claim_evidence_claim_id", "claim_evidence", ["claim_id"])
    op.create_check_constraint(
        RELATIONSHIP_CONSTRAINT,
        "claim_evidence",
        "relationship IN ('supports', 'contradicts', 'contextualizes')",
    )


def downgrade() -> None:
    op.drop_constraint(RELATIONSHIP_CONSTRAINT, "claim_evidence", type_="check")
    op.drop_table("claim_evidence")

    op.drop_constraint(STATUS_CONSTRAINT, "analysis_claims", type_="check")
    op.drop_constraint(CLAIM_TYPE_CONSTRAINT, "analysis_claims", type_="check")
    op.drop_table("analysis_claims")
