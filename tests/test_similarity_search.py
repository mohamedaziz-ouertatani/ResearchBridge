from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.embedding.search import find_similar_to_paper, search_by_text

MODEL = "test-model"


def _unit_vector(index: int, dim: int = EMBEDDING_DIM) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _blend(v1: list[float], v2: list[float], weight2: float) -> list[float]:
    combined = [(1 - weight2) * a + weight2 * b for a, b in zip(v1, v2, strict=True)]
    norm = sum(x * x for x in combined) ** 0.5
    return [x / norm for x in combined]


def _make_paper_with_embedding(session, source_id: str, vector: list[float]) -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source="fake",
        source_id=source_id,
        title=f"Paper {source_id}",
        abstract="abstract",
        raw_metadata={},
        ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    session.add(Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=MODEL, vector=vector))
    session.commit()
    return paper


@dataclass
class FixedVectorEmbedder:
    vector: list[float]
    model_name: str = MODEL

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def test_find_similar_orders_by_cosine_distance_ascending(session_factory) -> None:
    session = session_factory()
    target_vector = _unit_vector(0)
    close_vector = _blend(_unit_vector(0), _unit_vector(1), 0.1)  # nearly same direction as target
    orthogonal_vector = _unit_vector(1)  # 90 degrees off
    opposite_vector = [-x for x in _unit_vector(0)]  # 180 degrees off

    target = _make_paper_with_embedding(session, "target", target_vector)
    close = _make_paper_with_embedding(session, "close", close_vector)
    orthogonal = _make_paper_with_embedding(session, "orthogonal", orthogonal_vector)
    opposite = _make_paper_with_embedding(session, "opposite", opposite_vector)
    session.close()

    session = session_factory()
    try:
        results = find_similar_to_paper(session, target.id, MODEL, top_k=10)
        ordered_ids = [paper.id for paper, _distance in results]

        assert target.id not in ordered_ids  # the paper itself is excluded
        assert ordered_ids == [close.id, orthogonal.id, opposite.id]

        distances = [d for _paper, d in results]
        assert distances == sorted(distances)
        assert distances[0] < distances[1] < distances[2]
    finally:
        session.close()


def test_find_similar_respects_top_k(session_factory) -> None:
    session = session_factory()
    target = _make_paper_with_embedding(session, "target", _unit_vector(0))
    _make_paper_with_embedding(session, "b", _blend(_unit_vector(0), _unit_vector(1), 0.1))
    _make_paper_with_embedding(session, "c", _unit_vector(1))
    session.close()

    session = session_factory()
    try:
        results = find_similar_to_paper(session, target.id, MODEL, top_k=1)
        assert len(results) == 1
    finally:
        session.close()


def test_find_similar_raises_for_paper_without_embedding(session_factory) -> None:
    session = session_factory()
    paper = Paper(
        id=uuid.uuid4(),
        source="fake",
        source_id="no-embedding",
        title="No embedding",
        abstract="x",
        raw_metadata={},
        ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    session.close()

    session = session_factory()
    try:
        with pytest.raises(ValueError, match="No embedding found"):
            find_similar_to_paper(session, paper.id, MODEL, top_k=5)
    finally:
        session.close()


def test_search_by_text_embeds_query_and_finds_nearest(session_factory) -> None:
    session = session_factory()
    target_vector = _unit_vector(0)
    close = _make_paper_with_embedding(session, "close", _blend(_unit_vector(0), _unit_vector(1), 0.1))
    far = _make_paper_with_embedding(session, "far", _unit_vector(1))
    session.close()

    session = session_factory()
    try:
        embedder = FixedVectorEmbedder(vector=target_vector)
        results = search_by_text(session, "some query text", embedder, top_k=10)
        ordered_ids = [paper.id for paper, _distance in results]

        assert ordered_ids.index(close.id) < ordered_ids.index(far.id)
    finally:
        session.close()


def test_search_by_text_excludes_papers_marked_excluded(session_factory) -> None:
    session = session_factory()
    target = _unit_vector(0)
    included = _make_paper_with_embedding(session, "included", target)
    excluded = _make_paper_with_embedding(session, "excluded", target)
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = search_by_text(session, "query", FixedVectorEmbedder(target), top_k=10)

    result_ids = {paper.id for paper, _distance in results}
    assert included.id in result_ids
    assert excluded.id not in result_ids
    session.close()


def test_find_similar_to_paper_excludes_papers_marked_excluded(session_factory) -> None:
    session = session_factory()
    target = _unit_vector(0)
    seed = _make_paper_with_embedding(session, "seed", target)
    excluded = _make_paper_with_embedding(session, "excluded", target)
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = find_similar_to_paper(session, seed.id, MODEL, top_k=10)

    result_ids = {paper.id for paper, _distance in results}
    assert excluded.id not in result_ids
    session.close()
