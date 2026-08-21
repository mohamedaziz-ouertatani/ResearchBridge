"""Baseline 3 (Sec 27): sentence embeddings.

Thin adapter over the Week 6 pgvector similarity search (embedding/search.py)
so it fits the same Retriever protocol as the other baselines. fit() is a
no-op: the "index" is just the embeddings table, already built by rb-embed.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_by_text


class EmbeddingRetriever:
    name = "embedding"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._session: Session | None = None

    def fit(self, session: Session) -> None:
        self._session = session

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        if self._session is None:
            raise RuntimeError("call fit() before search()")

        # search_by_text ranks by cosine DISTANCE (0 = identical); Retriever
        # ranks by SCORE (higher = better), so invert to similarity.
        hits = search_by_text(self._session, query, self.embedder, top_k)
        return [(paper, 1.0 - distance) for paper, distance in hits]
