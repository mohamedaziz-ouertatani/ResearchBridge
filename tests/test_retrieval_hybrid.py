from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import Paper
from researchbridge.retrieval.hybrid import HybridRetriever


def _paper(source_id: str) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=source_id,
        raw_metadata={}, ingestion_metadata={},
    )


@dataclass
class FakeRetriever:
    """Returns a fixed, pre-ranked list regardless of the query - isolates RRF fusion logic from any real ranking algorithm."""

    name: str
    ranked_papers: list[Paper] = field(default_factory=list)
    fit_called: bool = False

    def fit(self, session) -> None:
        self.fit_called = True

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        # scores are irrelevant to RRF (rank-based), so fabricate decreasing values
        return [(p, 1.0 - i * 0.01) for i, p in enumerate(self.ranked_papers[:top_k])]


def test_fit_delegates_to_both_sub_retrievers() -> None:
    lexical, semantic = FakeRetriever("lex"), FakeRetriever("sem")
    hybrid = HybridRetriever(lexical, semantic)

    hybrid.fit(session=object())

    assert lexical.fit_called and semantic.fit_called


def test_doc_ranked_first_by_both_wins() -> None:
    a, b, c = _paper("a"), _paper("b"), _paper("c")
    lexical = FakeRetriever("lex", ranked_papers=[a, b, c])
    semantic = FakeRetriever("sem", ranked_papers=[a, c, b])

    hits = HybridRetriever(lexical, semantic).search("q", top_k=3)

    assert hits[0][0].source_id == "a"


def test_doc_present_in_only_one_ranker_still_appears() -> None:
    a, b = _paper("a"), _paper("b")
    lexical = FakeRetriever("lex", ranked_papers=[a])
    semantic = FakeRetriever("sem", ranked_papers=[b])

    hits = HybridRetriever(lexical, semantic).search("q", top_k=5)

    assert {p.source_id for p, _ in hits} == {"a", "b"}


def test_doc_agreed_on_by_both_outranks_doc_only_one_agrees_on() -> None:
    a, b, c = _paper("a"), _paper("b"), _paper("c")
    # b is #1 for lexical alone; a is #2 for both -> RRF should still favor
    # the doc both rankers agree is relevant over the doc only one ranks top.
    lexical = FakeRetriever("lex", ranked_papers=[b, a, c])
    semantic = FakeRetriever("sem", ranked_papers=[a, c, b])

    hits = HybridRetriever(lexical, semantic).search("q", top_k=3)

    assert hits[0][0].source_id == "a"


def test_top_k_limits_fused_results() -> None:
    papers = [_paper(str(i)) for i in range(10)]
    lexical = FakeRetriever("lex", ranked_papers=papers)
    semantic = FakeRetriever("sem", ranked_papers=list(reversed(papers)))

    hits = HybridRetriever(lexical, semantic, candidate_pool=10).search("q", top_k=3)

    assert len(hits) == 3
