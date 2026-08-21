from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.retrieval.tfidf import TfidfRetriever


def _paper(session, source_id: str, title: str, abstract: str = "") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    return paper


def test_ranks_lexically_matching_paper_first(session_factory) -> None:
    session = session_factory()
    _paper(session, "a", "Graph neural networks for fraud detection")
    _paper(session, "b", "Distributed consensus protocols")
    session.commit()

    retriever = TfidfRetriever()
    retriever.fit(session)
    hits = retriever.search("fraud detection", top_k=5)

    session.close()
    assert hits[0][0].source_id == "a"


def test_search_before_fit_raises(session_factory) -> None:
    import pytest

    retriever = TfidfRetriever()
    with pytest.raises(RuntimeError):
        retriever.search("anything", top_k=5)


def test_empty_corpus_returns_no_hits(session_factory) -> None:
    session = session_factory()
    retriever = TfidfRetriever()
    retriever.fit(session)

    hits = retriever.search("anything", top_k=5)

    session.close()
    assert hits == []


def test_zero_similarity_documents_are_excluded(session_factory) -> None:
    session = session_factory()
    _paper(session, "a", "Completely unrelated topic about gardening")
    session.commit()

    retriever = TfidfRetriever()
    retriever.fit(session)
    hits = retriever.search("quantum computing hardware", top_k=5)

    session.close()
    assert hits == []


def test_top_k_limits_results(session_factory) -> None:
    session = session_factory()
    for i in range(5):
        _paper(session, f"p{i}", f"Paper about neural networks number {i}")
    session.commit()

    retriever = TfidfRetriever()
    retriever.fit(session)
    hits = retriever.search("neural networks", top_k=2)

    session.close()
    assert len(hits) == 2
