from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.risks import assess_risks
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


def test_returns_none_for_empty_neighborhood(session) -> None:
    result = assess_risks(session, [])
    assert result.text is None
    assert result.evidence_ids == []


def test_returns_none_when_no_relevant_papers(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "requires substantial gpu resources")
    session.commit()

    result = assess_risks(session, [(paper.id, FAR)])

    assert result.text is None


def test_returns_none_when_no_limitations_claims_exist(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "a proposed method")
    session.commit()

    result = assess_risks(session, [(paper.id, NEAR)])

    assert result.text is None


def test_collects_a_limitation_from_a_relevant_paper(session) -> None:
    paper = _paper(session, "p1", title="Fraud Paper")
    evidence_id = _claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    result = assess_risks(session, [(paper.id, NEAR)])

    assert result.text is not None
    assert "evaluated only on offline datasets" in result.text
    assert "Fraud Paper" in result.text
    assert result.evidence_ids == [evidence_id]


def test_collects_limitations_from_multiple_relevant_papers_nearest_first(session) -> None:
    near_paper = _paper(session, "near", title="Near Paper")
    far_paper = _paper(session, "far", title="Far Paper")  # still within relevance gate, just farther
    e_near = _claim(session, near_paper, "limitations", "near limitation")
    e_far = _claim(session, far_paper, "limitations", "far limitation")
    session.commit()

    result = assess_risks(session, [(near_paper.id, 0.1), (far_paper.id, 0.4)])

    lines = result.text.split("\n")
    assert "near limitation" in lines[0]
    assert "far limitation" in lines[1]
    assert result.evidence_ids == [e_near, e_far]


def test_excludes_stub_claims(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "synthetic placeholder", extraction_method="stub")
    session.commit()

    result = assess_risks(session, [(paper.id, NEAR)])

    assert result.text is None


def test_excludes_too_distant_papers(session) -> None:
    near = _paper(session, "near")
    far = _paper(session, "far")
    e_near = _claim(session, near, "limitations", "a real limitation")
    _claim(session, far, "limitations", "an irrelevant limitation")
    session.commit()

    result = assess_risks(session, [(near.id, NEAR), (far.id, FAR)])

    assert "a real limitation" in result.text
    assert "an irrelevant limitation" not in result.text
    assert result.evidence_ids == [e_near]
