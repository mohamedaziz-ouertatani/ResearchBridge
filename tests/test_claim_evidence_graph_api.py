"""Route tests for /claim-graph and /gap-density (see test_claim_graph.py /
test_gap_density.py for the underlying builder unit tests)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    ClaimEvidence,
    Evidence,
    Paper,
    PaperCategory,
    ResearchAssessment,
    ResearchInput,
)


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.execute(
        text(
            "TRUNCATE TABLE claim_evidence, analysis_claims, candidate_gap_evidence, candidate_gaps, "
            "research_assessments, research_inputs CASCADE"
        )
    )
    s.commit()
    s.close()


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
    with TestClient(app) as c:
        yield c


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_evidence(session, paper: Paper, text_: str) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="results", section="results", text=text_,
        extraction_method="hybrid", model_version="v1", confidence="high",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _add_assessment_with_claim(session, candidate_gap_id=None):
    research_input = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(id=uuid.uuid4(), research_input_id=research_input.id, candidate_gap_id=candidate_gap_id)
    session.add(assessment)
    session.flush()
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="fact", claim_text="a claim", confidence="high", status="pending",
        source_table="research_assessments", source_id=assessment.id,
    )
    session.add(claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relationship="supports"))
    session.commit()
    return assessment


def test_claim_graph_returns_claims_and_evidence(client, session) -> None:
    assessment = _add_assessment_with_claim(session)

    response = client.get(f"/api/assessments/{assessment.id}/claim-graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 2
    assert {n["kind"] for n in body["nodes"]} == {"claim", "evidence"}
    assert len(body["edges"]) == 1
    assert body["edges"][0]["relationship"] == "supports"


def test_claim_graph_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/claim-graph")

    assert response.status_code == 404


def test_gap_density_returns_buckets(client, session) -> None:
    paper = _add_paper(session, "a paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=paper.id, observation="an observation", gap_type="inference",
        status="pending", contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.commit()
    assessment = _add_assessment_with_claim(session)

    response = client.get(f"/api/assessments/{assessment.id}/gap-density")

    assert response.status_code == 200
    body = response.json()
    assert body["buckets"] == [{"category": "Machine Learning", "gap_count": 1}]


def test_gap_density_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/gap-density")

    assert response.status_code == 404
