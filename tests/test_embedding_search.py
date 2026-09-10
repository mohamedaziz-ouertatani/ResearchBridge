from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper, PaperFullTextChunk
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.embedding.search import search_papers_with_fulltext


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _paper(session, source_id: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_abstract_embedding(session, embedder, paper, text) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )


def _add_chunk(session, embedder, paper, text, section="introduction", index=0) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        PaperFullTextChunk(
            paper_id=paper.id, section=section, paragraph_index=index, text=text,
            model_name=embedder.model_name, embedding=vector,
        )
    )


def test_surfaces_a_paper_that_only_matches_on_abstract(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert [p.id for p, _ in results] == [paper.id]
    session.close()


def test_surfaces_a_paper_that_only_matches_on_a_fulltext_chunk(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "unrelated abstract text")
    _add_chunk(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert [p.id for p, _ in results] == [paper.id]
    session.close()


def test_paper_with_only_abstract_still_surfaces(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert len(results) == 1
    session.close()


def test_excludes_curated_out_papers(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    paper.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert results == []
    session.close()


def test_ignores_chunks_from_a_different_model(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_chunk(session, FakeEmbedder(model_name="other-model"), paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert results == []
    session.close()
