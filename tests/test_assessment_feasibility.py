from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.feasibility import assess_technical_feasibility
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

NEAR = 0.1
FAR = 0.9


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
    result = assess_technical_feasibility(session, [])
    assert result.level == "not_assessed"
    assert result.evidence_ids == []


def test_not_assessed_when_no_relevant_papers(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "a proposed method")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, FAR)])

    assert result.level == "not_assessed"


def test_not_assessed_when_no_method_or_dataset_claims_exist(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "some limitation")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)])

    assert result.level == "not_assessed"


def test_medium_when_exactly_one_relevant_paper_has_grounding(session) -> None:
    paper = _paper(session, "p1", title="Method Paper")
    evidence_id = _claim(session, paper, "method", "a proposed method")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)])

    assert result.level == "medium"
    assert "Method Paper" in result.reasoning
    assert result.evidence_ids == [evidence_id]


def test_high_when_two_or_more_relevant_papers_have_grounding(session) -> None:
    a = _paper(session, "a", title="Paper A")
    b = _paper(session, "b", title="Paper B")
    e1 = _claim(session, a, "method", "method one")
    e2 = _claim(session, b, "dataset", "dataset two")
    session.commit()

    result = assess_technical_feasibility(session, [(a.id, NEAR), (b.id, NEAR)])

    assert result.level == "high"
    assert set(result.evidence_ids) == {e1, e2}


def test_excludes_stub_claims(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "synthetic placeholder", extraction_method="stub")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)])

    assert result.level == "not_assessed"


def test_excludes_too_distant_papers(session) -> None:
    near = _paper(session, "near", title="Near Paper")
    far = _paper(session, "far", title="Far Paper")
    e_near = _claim(session, near, "method", "a proposed method")
    _claim(session, far, "dataset", "an irrelevant dataset")
    session.commit()

    result = assess_technical_feasibility(session, [(near.id, NEAR), (far.id, FAR)])

    assert result.level == "medium"
    assert result.evidence_ids == [e_near]


def test_a_single_paper_with_both_method_and_dataset_still_counts_as_one_source(session) -> None:
    paper = _paper(session, "p1", title="Paper")
    e1 = _claim(session, paper, "method", "a proposed method")
    e2 = _claim(session, paper, "dataset", "a dataset")
    session.commit()

    result = assess_technical_feasibility(session, [(paper.id, NEAR)])

    assert result.level == "medium"  # one distinct paper, not two
    assert set(result.evidence_ids) == {e1, e2}


def test_reasoning_never_asserts_confident_engineering_feasibility(session) -> None:
    a = _paper(session, "a")
    b = _paper(session, "b")
    _claim(session, a, "method", "method one")
    _claim(session, b, "dataset", "dataset two")
    session.commit()

    result = assess_technical_feasibility(session, [(a.id, NEAR), (b.id, NEAR)])

    lowered = result.reasoning.lower()
    assert "will work" not in lowered
    assert "guaranteed" not in lowered
