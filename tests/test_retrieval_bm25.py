from __future__ import annotations

import uuid

import pytest

from researchbridge.db.models import Paper
from researchbridge.retrieval.bm25 import Bm25Retriever, tokenize


def _paper(session, source_id: str, title: str, abstract: str = "") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    return paper


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Graph-Based Fraud Detection!") == ["graph", "based", "fraud", "detection"]


def test_ranks_term_matching_paper_first(session_factory) -> None:
    # 3+ docs, not 2: with exactly 2 docs, any term appearing in exactly one
    # of them sits at document-frequency = N/2, where BM25's standard IDF
    # (log((N-n+0.5)/(n+0.5))) evaluates to log(1) = 0 for every such term -
    # a real property of the formula on tiny corpora, not a retriever bug.
    session = session_factory()
    _paper(session, "a", "Graph neural networks for fraud detection")
    _paper(session, "b", "Distributed consensus protocols")
    _paper(session, "c", "Reinforcement learning for robotic control")
    session.commit()

    retriever = Bm25Retriever()
    retriever.fit(session)
    hits = retriever.search("fraud detection", top_k=5)

    session.close()
    assert hits[0][0].source_id == "a"


def test_search_before_fit_raises(session_factory) -> None:
    retriever = Bm25Retriever()
    with pytest.raises(RuntimeError):
        retriever.search("anything", top_k=5)


def test_empty_corpus_returns_no_hits(session_factory) -> None:
    session = session_factory()
    retriever = Bm25Retriever()
    retriever.fit(session)

    hits = retriever.search("anything", top_k=5)

    session.close()
    assert hits == []


def test_no_term_overlap_returns_no_hits(session_factory) -> None:
    session = session_factory()
    _paper(session, "a", "Completely unrelated topic about gardening")
    session.commit()

    retriever = Bm25Retriever()
    retriever.fit(session)
    hits = retriever.search("quantum computing hardware", top_k=5)

    session.close()
    assert hits == []
