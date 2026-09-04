"""pid column on every *_runs table, for liveness detection

Revision ID: 0019
Revises: 0018
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RUN_TABLES = (
    "ingestion_runs",
    "extraction_runs",
    "embedding_runs",
    "citation_fetch_runs",
    "gap_detection_runs",
    "fulltext_fetch_runs",
)


def upgrade() -> None:
    for table in RUN_TABLES:
        op.add_column(table, sa.Column("pid", sa.Integer(), nullable=True))

    # Deliberately no data backfill here. A status="running" row that
    # predates this column has no pid to verify against - admin_routes.py's
    # liveness check treats that as "can't verify, so don't report it as
    # running" rather than guessing either way. Declaring every such row
    # "failed" as part of this migration would be a guess too: a run that
    # happens to still be genuinely in flight at the moment this migration
    # runs would get mislabeled failed while its process keeps writing to
    # a row the UI now says is dead. Cleaning up rows already known to be
    # abandoned (started days ago, clearly no process behind them) is a
    # separate, deliberate one-off documented in this migration's PR, not
    # something this migration does blindly for every row it finds.


def downgrade() -> None:
    for table in RUN_TABLES:
        op.drop_column(table, "pid")
