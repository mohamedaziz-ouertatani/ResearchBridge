"""add persistent Q&A collections and question snapshots

Revision ID: 0022
Revises: 0021
Create Date: 2026-09-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "qa_collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_qa_collections_updated_at", "qa_collections", ["updated_at"])

    op.create_table(
        "qa_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("hits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_citations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["qa_collections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_qa_questions_collection_id", "qa_questions", ["collection_id"])
    op.create_index("ix_qa_questions_created_at", "qa_questions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_qa_questions_created_at", table_name="qa_questions")
    op.drop_index("ix_qa_questions_collection_id", table_name="qa_questions")
    op.drop_table("qa_questions")
    op.drop_index("ix_qa_collections_updated_at", table_name="qa_collections")
    op.drop_table("qa_collections")
