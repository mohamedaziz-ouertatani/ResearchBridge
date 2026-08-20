"""Shared dependencies for the API.

Both the session factory and the embedder are overridable via FastAPI's
dependency_overrides, so tests can point at a test database and swap in a
fake embedder instead of loading the real ~80MB model.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session, sessionmaker

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


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
