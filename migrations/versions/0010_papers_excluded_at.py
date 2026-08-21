"""papers: add excluded_at (soft-delete for corpus curation)

NULL = included (default). Set = excluded from search, similar-papers,
ResearchAssessment retrieval, and gap-detection seed selection going
forward. Never a hard delete - see docs/superpowers/specs/
2026-08-21-corpus-curation-design.md for why.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("excluded_at", sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "excluded_at")
