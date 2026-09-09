"""add corpus_coverage_status (distinguishes an idea the corpus can actually
speak to from one where nothing relevant was retrieved at all - see
assessment/build.py's out-of-corpus suppression)

Revision ID: 0026
Revises: 0025
Create Date: 2026-09-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "research_assessments",
        sa.Column("corpus_coverage_status", sa.String(), nullable=False, server_default="not_assessed"),
    )
    op.alter_column("research_assessments", "corpus_coverage_status", server_default=None)


def downgrade() -> None:
    op.drop_column("research_assessments", "corpus_coverage_status")
