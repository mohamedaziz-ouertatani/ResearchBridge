"""Retriever interface: a ranked list of papers for a free-text query.

All four Sec 27 baselines (TF-IDF, BM25, embeddings, hybrid) implement
this, so the evaluation harness (Sec 27/28) can score any of them the same
way and a fifth method (a reranker, say) slots in without touching the
metrics code.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from researchbridge.db.models import Paper


class Retriever(Protocol):
    name: str

    def fit(self, session: Session) -> None:
        """Build the index over the current corpus. Call once before search()."""
        ...

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        """Return up to top_k (paper, score) pairs, highest-scoring first."""
        ...
