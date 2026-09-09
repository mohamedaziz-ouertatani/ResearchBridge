"""add context-scoped evidence reviews

Revision ID: 0024
Revises: 0023
Create Date: 2026-09-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("surface_type", sa.String(), nullable=False),
        sa.Column("surface_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "surface_type", "surface_id", "evidence_id", "role",
            name="uq_evidence_review_surface_evidence_role",
        ),
    )
    op.create_index("ix_evidence_reviews_surface", "evidence_reviews", ["surface_type", "surface_id"])


def downgrade() -> None:
    op.drop_index("ix_evidence_reviews_surface", table_name="evidence_reviews")
    op.drop_table("evidence_reviews")
