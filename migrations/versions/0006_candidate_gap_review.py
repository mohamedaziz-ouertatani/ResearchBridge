"""candidate_gaps: add reviewed_at, review_note (human review, Sec 35/44)

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("candidate_gaps", sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("candidate_gaps", sa.Column("review_note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("candidate_gaps", "review_note")
    op.drop_column("candidate_gaps", "reviewed_at")
