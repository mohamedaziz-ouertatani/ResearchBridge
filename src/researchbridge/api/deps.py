"""Shared dependencies for the API.

The session factory, embedder, and corpus IDF table are all overridable via
FastAPI's dependency_overrides, so tests can point at a test database, swap
in a fake embedder instead of loading the real ~80MB model, and swap in a
small fake IDF table instead of scanning the full corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

from researchbridge.assessment.claim_relevance import build_corpus_idf
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.model import SentenceTransformerEmbedder


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return make_session_factory(make_engine())


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """One process-wide embedder; the underlying model loads lazily on first use."""
    return SentenceTransformerEmbedder()


@lru_cache(maxsize=1)
def get_corpus_idf() -> dict[str, float]:
    """One process-wide IDF table for feasibility.py's widened-band claim-
    level overlap check (see that module's docstring). Built once, ~7-10s
    over the full corpus - far too slow to redo per request, same reasoning
    as get_embedder() above. Goes stale as new papers are ingested until the
    next process restart; no invalidation/refresh mechanism exists yet."""
    session = get_session_factory()()
    try:
        return build_corpus_idf(session)
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
