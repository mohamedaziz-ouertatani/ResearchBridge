from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.retrieval.embedding_retriever import EmbeddingRetriever


@dataclass
class FakeEmbedder:
    """Deterministic hash-based embedder - mirrors the one in test_embedding_pipeline.py."""

    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def _paper_with_embedding(session, embedder: FakeEmbedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector))
    return paper


def test_finds_the_paper_whose_embedding_matches_the_query(session_factory) -> None:
    embedder = FakeEmbedder()
    session = session_factory()
    _paper_with_embedding(session, embedder, "a", "Neural networks for vision")
    _paper_with_embedding(session, embedder, "b", "Distributed consensus protocols")
    session.commit()

    retriever = EmbeddingRetriever(embedder)
    retriever.fit(session)
    hits = retriever.search("Neural networks for vision", top_k=5)

    session.close()
    assert hits[0][0].source_id == "a"  # identical text embeds identically -> distance 0 -> score 1.0
    assert hits[0][1] == pytest.approx(1.0)


def test_search_before_fit_raises() -> None:
    retriever = EmbeddingRetriever(FakeEmbedder())
    with pytest.raises(RuntimeError):
        retriever.search("anything", top_k=5)
