from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.claim_relevance import CLAIM_OVERLAP_THRESHOLD, WIDENED_DISTANCE
from researchbridge.assessment.feasibility import assess_technical_feasibility
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

NEAR = 0.1
FAR = 0.9
WIDENED = 0.38  # inside (0.35, WIDENED_DISTANCE]
BEYOND_WIDENED = 0.45  # inside (WIDENED_DISTANCE, FAR) - must never be admitted

QUERY = "federated learning for sepsis prediction in icu patients"
NO_IDF: dict[str, float] = {}


def _paper(session, source_id: str, title: str = "a paper") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid") -> uuid.UUID:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )
    return evidence.id


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def test_not_assessed_for_empty_neighborhood(session) -> None:
    result = assess_technical_feasibility(session, [], QUERY, NO_IDF)
    assert result.level == "not_assessed"
    assert result.evidence_ids == []


def test_not_assessed_when_no_relevant_papers(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "a proposed method")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, FAR)], QUERY, NO_IDF)

    assert result.level == "not_assessed"


def test_not_assessed_when_no_method_or_dataset_claims_exist(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "some limitation")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)], QUERY, NO_IDF)

    assert result.level == "not_assessed"


def test_medium_when_exactly_one_relevant_paper_has_grounding(session) -> None:
    paper = _paper(session, "p1", title="Method Paper")
    evidence_id = _claim(session, paper, "method", "a proposed method")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)], QUERY, NO_IDF)

    assert result.level == "medium"
    assert "Method Paper" in result.reasoning
    assert result.evidence_ids == [evidence_id]


def test_high_when_two_or_more_relevant_papers_have_grounding(session) -> None:
    a = _paper(session, "a", title="Paper A")
    b = _paper(session, "b", title="Paper B")
    e1 = _claim(session, a, "method", "method one")
    e2 = _claim(session, b, "dataset", "dataset two")
    session.commit()

    result = assess_technical_feasibility(session, [(a.id, NEAR), (b.id, NEAR)], QUERY, NO_IDF)

    assert result.level == "high"
    assert set(result.evidence_ids) == {e1, e2}


def test_excludes_stub_claims(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "synthetic placeholder", extraction_method="stub")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)], QUERY, NO_IDF)

    assert result.level == "not_assessed"


def test_excludes_too_distant_papers(session) -> None:
    near = _paper(session, "near", title="Near Paper")
    far = _paper(session, "far", title="Far Paper")
    e_near = _claim(session, near, "method", "a proposed method")
    _claim(session, far, "dataset", "an irrelevant dataset")
    session.commit()

    result = assess_technical_feasibility(session, [(near.id, NEAR), (far.id, FAR)], QUERY, NO_IDF)

    assert result.level == "medium"
    assert result.evidence_ids == [e_near]


def test_a_single_paper_with_both_method_and_dataset_still_counts_as_one_source(session) -> None:
    paper = _paper(session, "p1", title="Paper")
    e1 = _claim(session, paper, "method", "a proposed method")
    e2 = _claim(session, paper, "dataset", "a dataset")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)], QUERY, NO_IDF)

    assert result.level == "medium"  # one distinct paper, not two
    assert set(result.evidence_ids) == {e1, e2}


def test_reasoning_never_asserts_confident_engineering_feasibility(session) -> None:
    a = _paper(session, "a")
    b = _paper(session, "b")
    _claim(session, a, "method", "method one")
    _claim(session, b, "dataset", "dataset two")
    session.commit()

    result = assess_technical_feasibility(session, [(a.id, NEAR), (b.id, NEAR)], QUERY, NO_IDF)

    lowered = result.reasoning.lower()
    assert "will work" not in lowered
    assert "guaranteed" not in lowered


# --- widened-band (0.35 < distance <= WIDENED_DISTANCE) regression tests ---


def test_widened_paper_admitted_when_overlap_meets_threshold(session) -> None:
    paper = _paper(session, "p1", title="Widened Paper")
    evidence_id = _claim(session, paper, "method", "a federated learning approach for sepsis in icu patients")
    session.commit()

    idf = {"federated": 3.0, "learning": 1.0, "sepsis": 3.0, "icu": 3.0, "patients": 2.0}

    result = assess_technical_feasibility(session, [(paper.id, WIDENED)], QUERY, idf)

    assert result.level == "medium"
    assert result.evidence_ids == [evidence_id]


def test_widened_paper_excluded_when_overlap_below_threshold(session) -> None:
    paper = _paper(session, "p1", title="Generic Widened Paper")
    _claim(session, paper, "method", "an unrelated deep learning framework for image classification")
    session.commit()

    idf = {"federated": 3.0, "sepsis": 3.0, "icu": 3.0}  # none of these appear in the claim

    result = assess_technical_feasibility(session, [(paper.id, WIDENED)], QUERY, idf)

    assert result.level == "not_assessed"
    assert result.evidence_ids == []


def test_paper_beyond_widened_distance_excluded_regardless_of_overlap(session) -> None:
    paper = _paper(session, "p1", title="Too Far Paper")
    _claim(session, paper, "method", "federated learning for sepsis prediction in icu patients")
    session.commit()

    # identical text to the query -> maximum possible overlap - still must be excluded
    idf = {"federated": 3.0, "sepsis": 3.0, "icu": 3.0, "patients": 2.0}

    result = assess_technical_feasibility(session, [(paper.id, BEYOND_WIDENED)], QUERY, idf)

    assert result.level == "not_assessed"
    assert result.evidence_ids == []


def test_widened_paper_stub_claim_still_excluded_even_with_high_overlap(session) -> None:
    paper = _paper(session, "p1", title="Stub Widened Paper")
    _claim(
        session, paper, "method", "federated learning for sepsis prediction in icu patients",
        extraction_method="stub",
    )
    session.commit()

    idf = {"federated": 3.0, "sepsis": 3.0, "icu": 3.0, "patients": 2.0}

    result = assess_technical_feasibility(session, [(paper.id, WIDENED)], QUERY, idf)

    assert result.level == "not_assessed"


def test_widened_paper_admitted_via_max_overlap_across_multiple_claims(session) -> None:
    paper = _paper(session, "p1", title="Mixed Claims Paper")
    _claim(session, paper, "dataset", "an unrelated benchmark dataset")  # low overlap
    e_strong = _claim(session, paper, "method", "federated learning for sepsis prediction in icu patients")  # high overlap

    session.commit()

    idf = {"federated": 3.0, "sepsis": 3.0, "icu": 3.0, "patients": 2.0}

    result = assess_technical_feasibility(session, [(paper.id, WIDENED)], QUERY, idf)

    assert result.level == "medium"
    assert result.evidence_ids == [e_strong]


def test_widened_paper_does_not_change_existing_close_paper_result(session) -> None:
    """A widened-band paper that fails the overlap check must not affect an
    already-admitted close (<=0.35) paper's result at all - the two paths
    are independent, not sequential filters on the same set."""
    close = _paper(session, "close", title="Close Paper")
    widened = _paper(session, "widened", title="Widened Paper")
    e_close = _claim(session, close, "method", "a proposed method")
    _claim(session, widened, "method", "an unrelated deep learning framework for image classification")
    session.commit()

    idf = {"federated": 3.0, "sepsis": 3.0}  # nothing matches the widened claim

    result = assess_technical_feasibility(session, [(close.id, NEAR), (widened.id, WIDENED)], QUERY, idf)

    assert result.level == "medium"
    assert result.evidence_ids == [e_close]


def test_reasoning_distinguishes_widened_admission_from_close_admission(session) -> None:
    close = _paper(session, "close", title="Close Paper")
    widened = _paper(session, "widened", title="Widened Paper")
    _claim(session, close, "method", "a proposed method")
    _claim(session, widened, "method", "federated learning for sepsis prediction in icu patients")
    session.commit()

    idf = {"federated": 3.0, "sepsis": 3.0, "icu": 3.0, "patients": 2.0}

    result = assess_technical_feasibility(session, [(close.id, NEAR), (widened.id, WIDENED)], QUERY, idf)

    assert result.level == "high"
    assert "Widened Paper" in result.reasoning
    assert "shared technical" in result.reasoning.lower() or "wider match" in result.reasoning.lower()


def test_calibrated_constants_are_the_expected_values() -> None:
    assert WIDENED_DISTANCE == 0.40
    assert CLAIM_OVERLAP_THRESHOLD == 0.20
