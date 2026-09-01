"""candidate_gaps/candidate_gap_evidence: gap_status, resolution_note, and
per-evidence classification provenance

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

GAP_STATUS_CONSTRAINT = "ck_candidate_gaps_gap_status"
CLAIM_ROLE_CONSTRAINT = "ck_candidate_gap_evidence_claim_role"


def upgrade() -> None:
    op.add_column("candidate_gaps", sa.Column("gap_status", sa.String(), nullable=True))
    op.add_column("candidate_gaps", sa.Column("resolution_note", sa.Text(), nullable=True))
    op.create_check_constraint(
        GAP_STATUS_CONSTRAINT,
        "candidate_gaps",
        "gap_status IS NULL OR gap_status IN ('strong_gap', 'potential_gap', 'known_limitation')",
    )

    op.add_column("candidate_gap_evidence", sa.Column("claim_role", sa.String(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("self_resolution_signal", sa.Boolean(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("field_scope_signal", sa.Boolean(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("own_contribution_overlap", sa.Float(), nullable=True))
    op.create_check_constraint(
        CLAIM_ROLE_CONSTRAINT,
        "candidate_gap_evidence",
        "claim_role IS NULL OR claim_role IN ('anchor', 'supporting')",
    )


def downgrade() -> None:
    op.drop_constraint(CLAIM_ROLE_CONSTRAINT, "candidate_gap_evidence", type_="check")
    op.drop_column("candidate_gap_evidence", "own_contribution_overlap")
    op.drop_column("candidate_gap_evidence", "field_scope_signal")
    op.drop_column("candidate_gap_evidence", "self_resolution_signal")
    op.drop_column("candidate_gap_evidence", "claim_role")

    op.drop_constraint(GAP_STATUS_CONSTRAINT, "candidate_gaps", type_="check")
    op.drop_column("candidate_gaps", "resolution_note")
    op.drop_column("candidate_gaps", "gap_status")
