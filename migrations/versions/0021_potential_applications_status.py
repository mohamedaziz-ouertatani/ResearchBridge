"""add potential_applications_status (distinguishes "no relevant papers
retrieved" from "relevant papers retrieved but none stated an application" -
see assessment/applications.py's ApplicationsResult.status)

Revision ID: 0021
Revises: 0020
Create Date: 2026-09-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "research_assessments",
        sa.Column("potential_applications_status", sa.String(), nullable=False, server_default="not_assessed"),
    )
    op.alter_column("research_assessments", "potential_applications_status", server_default=None)


def downgrade() -> None:
    op.drop_column("research_assessments", "potential_applications_status")
