"""add research project workspaces

Revision ID: 0023
Revises: 0022
Create Date: 2026-09-06
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "research_projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_projects_updated_at", "research_projects", ["updated_at"])

    op.create_table(
        "research_project_inputs",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_input_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["research_projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["research_input_id"], ["research_inputs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id", "research_input_id"),
        sa.UniqueConstraint("project_id", "research_input_id", name="uq_research_project_input"),
    )
    op.create_index("ix_research_project_inputs_research_input_id", "research_project_inputs", ["research_input_id"])


def downgrade() -> None:
    op.drop_index("ix_research_project_inputs_research_input_id", table_name="research_project_inputs")
    op.drop_table("research_project_inputs")
    op.drop_index("ix_research_projects_updated_at", table_name="research_projects")
    op.drop_table("research_projects")
