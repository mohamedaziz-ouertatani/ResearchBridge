"""research_assessments: add technical_feasibility_reasoning

Mirrors novelty_level/novelty_reasoning's existing pattern - a bare
categorical level with no explanation reads as unexplained certainty.

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("research_assessments", sa.Column("technical_feasibility_reasoning", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("research_assessments", "technical_feasibility_reasoning")
