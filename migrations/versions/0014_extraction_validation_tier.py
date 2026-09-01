"""extracted_claims: add validation_tier

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("extracted_claims", sa.Column("validation_tier", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("extracted_claims", "validation_tier")
