"""add nearest_distance / mean_distance (the two retrieval numbers
corpus_coverage_status is decided from, persisted so the verdict is
auditable in the report and export layers, which have no embedder)

Revision ID: 0027
Revises: 0026
Create Date: 2026-09-09
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("research_assessments", sa.Column("nearest_distance", sa.Float(), nullable=True))
    op.add_column("research_assessments", sa.Column("mean_distance", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("research_assessments", "mean_distance")
    op.drop_column("research_assessments", "nearest_distance")
