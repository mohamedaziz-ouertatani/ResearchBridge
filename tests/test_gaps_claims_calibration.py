from __future__ import annotations

import uuid

from researchbridge.db.models import AnalysisClaim, CandidateGap, Paper
from researchbridge.gaps.claims_calibration import gap_claim_confidence_vs_ratings


def _paper(session) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id="p1", title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _gap(
    session,
    paper: Paper,
    status: str = "approved",
    correctness_rating: int | None = None,
    usefulness_rating: int | None = None,
) -> CandidateGap:
    gap = CandidateGap(
        seed_paper_id=paper.id, observation="obs", gap_type="inference", status=status,
        contributing_paper_count=3, similarity_threshold=0.5, detection_method="cluster-v2",
        correctness_rating=correctness_rating, usefulness_rating=usefulness_rating,
    )
    session.add(gap)
    session.flush()
    return gap


def _claim(session, gap: CandidateGap, confidence: str = "medium") -> AnalysisClaim:
    claim = AnalysisClaim(
        claim_type="inference", claim_text="obs", confidence=confidence, status=gap.status,
        source_table="candidate_gaps", source_id=gap.id,
    )
    session.add(claim)
    session.commit()
    return claim


def test_groups_reviewed_gaps_by_confidence_bucket(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    gap1 = _gap(session, paper, status="approved", correctness_rating=3)
    gap2 = _gap(session, paper, status="approved", correctness_rating=1)
    _claim(session, gap1, confidence="medium")
    _claim(session, gap2, confidence="medium")

    stats = gap_claim_confidence_vs_ratings(session)
    session.close()

    assert len(stats) == 1
    assert stats[0].confidence == "medium"
    assert stats[0].sample_count == 2
    assert stats[0].mean_ratings["correctness_rating"] == 2.0


def test_separates_distinct_confidence_buckets(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    gap_high = _gap(session, paper, status="approved", correctness_rating=3)
    gap_low = _gap(session, paper, status="rejected", correctness_rating=0)
    _claim(session, gap_high, confidence="high")
    _claim(session, gap_low, confidence="low")

    stats = gap_claim_confidence_vs_ratings(session)
    session.close()

    by_confidence = {s.confidence: s for s in stats}
    assert by_confidence["high"].mean_ratings["correctness_rating"] == 3.0
    assert by_confidence["low"].mean_ratings["correctness_rating"] == 0.0


def test_pending_gaps_are_excluded(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    pending_gap = _gap(session, paper, status="pending", correctness_rating=3)
    _claim(session, pending_gap, confidence="medium")

    stats = gap_claim_confidence_vs_ratings(session)
    session.close()

    assert stats == []


def test_omits_rating_dimensions_with_no_values_set(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    gap = _gap(session, paper, status="approved", correctness_rating=None, usefulness_rating=2)
    _claim(session, gap, confidence="medium")

    stats = gap_claim_confidence_vs_ratings(session)
    session.close()

    assert "correctness_rating" not in stats[0].mean_ratings
    assert stats[0].mean_ratings["usefulness_rating"] == 2.0


def test_no_reviewed_gaps_returns_empty_list(session_factory) -> None:
    session = session_factory()
    stats = gap_claim_confidence_vs_ratings(session)
    session.close()
    assert stats == []
