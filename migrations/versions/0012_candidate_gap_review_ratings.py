"""candidate_gaps: add five human review rating dimensions (Sec 44)

correctness/relevance/novelty/evidence_support/usefulness, each 0-3 per
Sec 44's rubric (0=irrelevant, 1=weak, 2=plausible, 3=highly relevant) -
nullable, same as review_note, since a reviewer can approve/reject without
rating (rating is an enrichment of the existing binary review, not a
replacement for it).

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-29
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RATING_COLUMNS = (
    "correctness_rating",
    "relevance_rating",
    "novelty_rating",
    "evidence_support_rating",
    "usefulness_rating",
)


def upgrade() -> None:
    for column in RATING_COLUMNS:
        op.add_column("candidate_gaps", sa.Column(column, sa.Integer(), nullable=True))
        op.create_check_constraint(
            f"ck_candidate_gaps_{column}_range",
            "candidate_gaps",
            f"{column} IS NULL OR {column} BETWEEN 0 AND 3",
        )


def downgrade() -> None:
    for column in reversed(RATING_COLUMNS):
        op.drop_constraint(f"ck_candidate_gaps_{column}_range", "candidate_gaps", type_="check")
        op.drop_column("candidate_gaps", column)
