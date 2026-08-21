from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.opportunities import assess_opportunities
from researchbridge.db.models import Evidence, ExtractedClaim, Paper


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def _paper(session, source_id: str, title: str = "a paper") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def test_always_returns_none_even_with_a_strong_application_present(session) -> None:
    paper = _paper(session, "p1", title="Fraud Paper")
    evidence = Evidence(
        paper_id=paper.id, evidence_type="applications", section=None, text="real-time fraud screening",
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(
            paper_id=paper.id, claim_type="applications", text="real-time fraud screening",
            evidence_id=evidence.id, confidence="medium",
        )
    )
    session.commit()

    result = assess_opportunities(session, [(paper.id, 0.1)])

    assert result.opportunities is None
    assert result.evidence_ids == []


def test_returns_none_for_empty_neighborhood(session) -> None:
    result = assess_opportunities(session, [])

    assert result.opportunities is None
    assert result.evidence_ids == []
