from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from researchbridge.db.models import EMBEDDING_DIM, Embedding, EmbeddingRun, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE, EmbeddingPipeline


def _make_paper(session, source_id: str, title: str = "A Paper", abstract: str | None = "An abstract.") -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source="fake",
        source_id=source_id,
        title=title,
        abstract=abstract,
        raw_metadata={},
        ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


@dataclass
class FakeEmbedder:
    """Deterministic, fast fake: hashes each text into a fixed-size unit vector.

    Never loads the real ~80MB sentence-transformers model, so pipeline
    mechanics tests stay fast and independent of it.
    """

    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(text) for text in texts]


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def test_embeds_paper_with_title_and_abstract(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", title="Title Here", abstract="Abstract here.")
    session.close()

    embedder = FakeEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(EmbeddingRun, run_id)
        assert run.status == "completed"
        assert run.papers_processed == 1
        assert run.papers_skipped == 0

        embedding = session.execute(select(Embedding).where(Embedding.paper_id == paper.id)).scalar_one()
        assert embedding.embedding_type == EMBEDDING_TYPE
        assert embedding.model_name == "fake-embedder-v1"
        assert len(embedding.vector) == EMBEDDING_DIM

        assert embedder.calls == [["Title Here\n\nAbstract here."]]
    finally:
        session.close()


def test_embeds_paper_with_missing_abstract_using_title_only(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1", title="Title Only", abstract=None)
    session.close()

    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(), session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(EmbeddingRun, run_id)
        assert run.papers_processed == 1
        assert run.papers_skipped == 0
        assert len(session.execute(select(Embedding)).scalars().all()) == 1
    finally:
        session.close()


def test_embedding_is_idempotent_across_runs(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1")
    session.close()

    embedder = FakeEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, session_factory=session_factory)
    first_run_id = pipeline.run()
    second_run_id = pipeline.run()

    session = session_factory()
    try:
        first_run = session.get(EmbeddingRun, first_run_id)
        second_run = session.get(EmbeddingRun, second_run_id)
        assert first_run.papers_processed == 1
        assert second_run.papers_processed == 0  # nothing left to embed

        assert len(session.execute(select(Embedding)).scalars().all()) == 1  # not duplicated
    finally:
        session.close()


def test_different_model_name_creates_separate_embedding_row(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1")
    session.close()

    EmbeddingPipeline(embedder=FakeEmbedder(model_name="model-a"), session_factory=session_factory).run()
    EmbeddingPipeline(embedder=FakeEmbedder(model_name="model-b"), session_factory=session_factory).run()

    session = session_factory()
    try:
        rows = session.execute(select(Embedding).where(Embedding.paper_id == paper.id)).scalars().all()
        assert {r.model_name for r in rows} == {"model-a", "model-b"}
    finally:
        session.close()


def test_limit_caps_papers_processed(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1")
    _make_paper(session, "P2")
    _make_paper(session, "P3")
    session.close()

    pipeline = EmbeddingPipeline(embedder=FakeEmbedder(), session_factory=session_factory)
    run_id = pipeline.run(limit=2)

    session = session_factory()
    try:
        run = session.get(EmbeddingRun, run_id)
        assert run.papers_processed == 2
    finally:
        session.close()


def test_batches_across_batch_size_boundary(session_factory) -> None:
    session = session_factory()
    for i in range(35):  # > BATCH_SIZE (32), exercises the chunking loop
        _make_paper(session, f"P{i}")
    session.close()

    embedder = FakeEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(EmbeddingRun, run_id)
        assert run.papers_processed == 35
        assert len(session.execute(select(Embedding)).scalars().all()) == 35
        assert len(embedder.calls) == 2  # two batches: 32 + 3
    finally:
        session.close()
