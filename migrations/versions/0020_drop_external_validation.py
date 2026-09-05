"""drop external_validation_needed (feature removed - see git history)

Revision ID: 0020
Revises: 0019
Create Date: 2026-09-05
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("research_assessments", "external_validation_needed")


def downgrade() -> None:
    op.add_column("research_assessments", sa.Column("external_validation_needed", sa.Text(), nullable=True))
