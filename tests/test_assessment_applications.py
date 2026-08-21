from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.applications import assess_applications
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
    result = assess_applications(session, [])
    assert result.applications is None
    assert result.evidence_ids == []


def test_returns_none_when_no_relevant_papers(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "applications", "fraud detection")
    session.commit()

    result = assess_applications(session, [(paper.id, FAR)])

    assert result.applications is None


def test_returns_none_when_no_applications_claims_exist(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "some limitation")
    session.commit()

    result = assess_applications(session, [(paper.id, NEAR)])

    assert result.applications is None


def test_collects_application_from_a_relevant_paper(session) -> None:
    paper = _paper(session, "p1", title="Fraud Paper")
    evidence_id = _claim(session, paper, "applications", "real-time payment fraud screening")
    session.commit()

    result = assess_applications(session, [(paper.id, NEAR)])

    assert result.applications is not None
    assert len(result.applications) == 1
    app = result.applications[0]
    assert app.application == "real-time payment fraud screening"
    assert app.source_paper == "Fraud Paper"
    assert app.paper_id == paper.id
    assert result.evidence_ids == [evidence_id]


def test_collects_applications_from_multiple_relevant_papers_nearest_first(session) -> None:
    near_paper = _paper(session, "near", title="Near Paper")
    far_paper = _paper(session, "far", title="Far Paper")  # still within relevance gate, just farther
    e_near = _claim(session, near_paper, "applications", "near application")
    e_far = _claim(session, far_paper, "applications", "far application")
    session.commit()

    result = assess_applications(session, [(near_paper.id, 0.1), (far_paper.id, 0.4)])

    assert [a.application for a in result.applications] == ["near application", "far application"]
    assert result.evidence_ids == [e_near, e_far]


def test_excludes_stub_claims(session) -> None:
    paper = _paper(session, "p1")
    _claim(session, paper, "applications", "synthetic placeholder", extraction_method="stub")
    session.commit()

    result = assess_applications(session, [(paper.id, NEAR)])

    assert result.applications is None


def test_excludes_too_distant_papers(session) -> None:
    near_paper = _paper(session, "near")
    far_paper = _paper(session, "far")
    e_near = _claim(session, near_paper, "applications", "a real application")
    _claim(session, far_paper, "applications", "an irrelevant application")
    session.commit()

    result = assess_applications(session, [(near_paper.id, NEAR), (far_paper.id, FAR)])

    assert len(result.applications) == 1
    assert result.applications[0].application == "a real application"
    assert result.evidence_ids == [e_near]
