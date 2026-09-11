"""embeddings HNSW cosine index (assessment-pipeline perf pass)

embedding/search.py's _nearest() has done a sequential scan over every row
in `embeddings` on every retrieval since Phase 1, deliberately deferred
until corpus size demonstrated it was needed (see that module's own
docstring). The corpus is now 82k+ papers/embeddings, all sharing a single
(embedding_type, model_name) pair, and every assessment build calls
retrieval at least once - HNSW (pgvector >=0.5, confirmed 0.8.6 here) on
`vector` with cosine ops turns that scan into an approximate-nearest-
neighbor lookup, with build.py's `_nearest()`/`search_by_text()` queries
unchanged since the operator class matches the same `<=>` cosine-distance
operator already in use.

Revision ID: 0028
Revises: 0027
Create Date: 2026-09-11
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0028"
down_revision: Union[str, None] = "0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # CONCURRENTLY so the (82k-row, growing) embeddings table stays
    # writable/readable for the duration of the build - a plain CREATE
    # INDEX takes a lock that blocks concurrent ingestion writes.
    # CONCURRENTLY can't run inside a transaction, hence the autocommit
    # block (Alembic normally wraps each migration in one).
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_embeddings_vector_hnsw_cosine "
            "ON embeddings USING hnsw (vector vector_cosine_ops)"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_embeddings_vector_hnsw_cosine")
