"""add potential_opportunities_status (distinguishes "no qualifying
applications" from "Ollama unavailable/synthesis failed" from "found" -
see assessment/opportunity_synthesis.py)

Revision ID: 0025
Revises: 0024
Create Date: 2026-09-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "research_assessments",
        sa.Column("potential_opportunities_status", sa.String(), nullable=False, server_default="not_assessed"),
    )
    op.alter_column("research_assessments", "potential_opportunities_status", server_default=None)


def downgrade() -> None:
    op.drop_column("research_assessments", "potential_opportunities_status")
