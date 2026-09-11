"""Similarity search over paper embeddings, using pgvector cosine distance.

`embeddings.vector` is backed by an HNSW index (migration 0028, added once
the corpus reached 82k+ papers and a sequential scan was no longer "fine
at Phase 1's scale" - see that migration's docstring). ANN, so results are
approximate nearest-neighbor rather than an exact sort, same tradeoff
every other cosine_distance() ORDER BY ... LIMIT query in this module now
makes.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.db.models import Embedding, Paper, PaperFullTextChunk
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def find_similar_to_paper(
    session: Session, paper_id: uuid.UUID, model_name: str, top_k: int = 10
) -> list[tuple[Paper, float]]:
    target = session.execute(
        select(Embedding.vector).where(
            Embedding.paper_id == paper_id,
            Embedding.embedding_type == EMBEDDING_TYPE,
            Embedding.model_name == model_name,
        )
    ).scalar_one_or_none()
    if target is None:
        raise ValueError(f"No embedding found for paper {paper_id} (model={model_name})")

    return _nearest(session, target, model_name, top_k, exclude_paper_id=paper_id)


def search_by_text(session: Session, query_text: str, embedder: Embedder, top_k: int = 10) -> list[tuple[Paper, float]]:
    [vector] = embedder.embed_texts([query_text])
    return _nearest(session, vector, embedder.model_name, top_k, exclude_paper_id=None)


def _nearest(
    session: Session,
    vector: list[float],
    model_name: str,
    top_k: int,
    exclude_paper_id: uuid.UUID | None,
) -> list[tuple[Paper, float]]:
    distance = Embedding.vector.cosine_distance(vector).label("distance")
    query = (
        select(Paper, distance)
        .join(Embedding, Embedding.paper_id == Paper.id)
        .where(
            Embedding.embedding_type == EMBEDDING_TYPE,
            Embedding.model_name == model_name,
            Paper.excluded_at.is_(None),
        )
    )
    if exclude_paper_id is not None:
        query = query.where(Paper.id != exclude_paper_id)
    query = query.order_by(distance.asc()).limit(top_k)

    return [(row[0], row[1]) for row in session.execute(query).all()]


def search_papers_with_fulltext(
    session: Session, query_text: str, embedder: Embedder, top_k: int = 10
) -> list[tuple[Paper, float]]:
    """Like search_by_text, but a paper can also surface via its
    best-matching full-text paragraph, not just its title/abstract
    embedding - a paper's score is whichever of the two is closer (LEAST:
    smaller cosine distance = more similar). A paper missing either side
    (no abstract embedding yet, or no full-text chunks) still surfaces on
    whichever side it has - COALESCE treats the missing side as maximally
    far (2.0, the max possible cosine distance) rather than excluding the
    paper.
    """
    [vector] = embedder.embed_texts([query_text])

    abstract_distance = Embedding.vector.cosine_distance(vector)
    best_chunk = (
        select(
            PaperFullTextChunk.paper_id.label("paper_id"),
            func.min(PaperFullTextChunk.embedding.cosine_distance(vector)).label("distance"),
        )
        .where(PaperFullTextChunk.model_name == embedder.model_name)
        .group_by(PaperFullTextChunk.paper_id)
        .subquery()
    )

    combined = func.least(
        func.coalesce(abstract_distance, 2.0), func.coalesce(best_chunk.c.distance, 2.0)
    ).label("combined")

    query = (
        select(Paper, combined)
        .select_from(Paper)
        .outerjoin(
            Embedding,
            (Embedding.paper_id == Paper.id)
            & (Embedding.embedding_type == EMBEDDING_TYPE)
            & (Embedding.model_name == embedder.model_name),
        )
        .outerjoin(best_chunk, best_chunk.c.paper_id == Paper.id)
        .where(
            Paper.excluded_at.is_(None),
            (Embedding.id.isnot(None)) | (best_chunk.c.paper_id.isnot(None)),
        )
        .order_by(combined.asc())
        .limit(top_k)
    )
    return [(row[0], row[1]) for row in session.execute(query).all()]
