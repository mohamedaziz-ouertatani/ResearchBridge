"""candidate_gaps: enforce status in (pending, approved, rejected) at the DB level

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT_NAME = "ck_candidate_gaps_status"


def upgrade() -> None:
    op.create_check_constraint(
        CONSTRAINT_NAME,
        "candidate_gaps",
        "status IN ('pending', 'approved', 'rejected')",
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "candidate_gaps", type_="check")
