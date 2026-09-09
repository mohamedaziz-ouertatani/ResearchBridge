from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import Evidence, ResearchAssessment, ResearchAssessmentEvidence, ResearchInput


@pytest.fixture()
def client(session_factory):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    with TestClient(app) as test_client:
        yield test_client


def test_review_evidence_is_idempotent_and_scoped_to_assessment_role(client, session_factory) -> None:
    session = session_factory()
    research_input = ResearchInput(input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(research_input_id=research_input.id, status="completed")
    evidence = Evidence(
        paper_id=uuid.uuid4(), evidence_type="method", section="methods", text="quote",
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    # The paper FK is intentionally satisfied below with a real paper row.
    from researchbridge.db.models import Paper
    paper = Paper(id=evidence.paper_id, source="arxiv", source_id="review", title="Paper", abstract="", raw_metadata={}, ingestion_metadata={})
    session.add(paper)
    session.flush()
    session.add_all([assessment, evidence])
    session.flush()
    session.add(ResearchAssessmentEvidence(research_assessment_id=assessment.id, evidence_id=evidence.id, role="novelty"))
    session.commit()
    session.close()

    payload = {
        "evidence_id": str(evidence.id),
        "surface_type": "assessment",
        "surface_id": str(assessment.id),
        "role": "novelty",
        "verdict": "supported",
        "note": "Directly supports the comparison.",
    }
    first = client.put("/api/evidence/review", json=payload)
    second = client.put("/api/evidence/review", json={**payload, "verdict": "unclear"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["verdict"] == "unclear"
    assert second.json()["note"] == payload["note"]


def test_review_rejects_evidence_not_linked_to_assessment_role(client, session_factory) -> None:
    session = session_factory()
    from researchbridge.db.models import Paper
    paper = Paper(id=uuid.uuid4(), source="arxiv", source_id="unlinked", title="Paper", abstract="", raw_metadata={}, ingestion_metadata={})
    session.add(paper)
    evidence = Evidence(paper_id=paper.id, evidence_type="method", text="quote", extraction_method="hybrid", model_version="v1", confidence="medium")
    research_input = ResearchInput(input_type="idea", raw_text="an idea")
    session.add_all([paper, research_input])
    session.flush()
    session.add(evidence)
    session.flush()
    assessment = ResearchAssessment(research_input_id=research_input.id, status="completed")
    session.add(assessment)
    session.commit()
    session.close()

    response = client.put("/api/evidence/review", json={
        "evidence_id": str(evidence.id),
        "surface_type": "assessment",
        "surface_id": str(assessment.id),
        "role": "novelty",
        "verdict": "not_supported",
    })
    assert response.status_code == 422
