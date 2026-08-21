"""Similarity search over paper embeddings, using pgvector cosine distance.

No ANN index (ivfflat/hnsw) yet — a sequential scan is fine at Phase 1's
corpus scale. Add one only once corpus size demonstrates it's needed, same
reasoning as deferring a dedicated graph database (blueprint §12/§47).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Embedding, Paper
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
