"""extraction_runs/embedding_runs: add force (was this a --force re-run)

Purely informational - lets the notifications feed (and anyone reading
run history) tell a normal incremental run apart from one that wiped
prior results first. Does not affect reset behavior itself, which
still happens in the CLI before the run row exists.

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-25
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "extraction_runs", sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.add_column(
        "embedding_runs", sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    op.drop_column("embedding_runs", "force")
    op.drop_column("extraction_runs", "force")
