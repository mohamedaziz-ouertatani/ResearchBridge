"""authors, paper_authors, paper_categories, paper_citations

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "authors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("orcid", sa.String(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("name", name="uq_authors_name"),
    )

    op.create_table(
        "paper_authors",
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), primary_key=True),
        sa.Column("author_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("authors.id"), primary_key=True),
        sa.Column("author_order", sa.Integer(), nullable=False),
    )

    op.create_table(
        "paper_categories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.UniqueConstraint(
            "paper_id", "category", "source", name="uq_paper_categories_paper_category_source"
        ),
    )
    op.create_index("ix_paper_categories_paper_id", "paper_categories", ["paper_id"])

    op.create_table(
        "paper_citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("citing_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("cited_paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("confidence", sa.String(), nullable=False),
        sa.UniqueConstraint(
            "citing_paper_id", "cited_paper_id", "source", name="uq_paper_citations_edge_source"
        ),
    )
    op.create_index("ix_paper_citations_citing_paper_id", "paper_citations", ["citing_paper_id"])
    op.create_index("ix_paper_citations_cited_paper_id", "paper_citations", ["cited_paper_id"])


def downgrade() -> None:
    op.drop_table("paper_citations")
    op.drop_table("paper_categories")
    op.drop_table("paper_authors")
    op.drop_table("authors")
