from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import AnalysisClaim, ClaimEvidence, Evidence, ExtractedClaim, Paper


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


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    # claim_evidence/analysis_claims aren't in conftest's TRUNCATE list, and
    # other test files (test_gaps_claims.py, test_assessment_claims.py) add
    # rows without cleaning them up - truncate before too, not just after,
    # so this file's unfiltered total counts aren't polluted by leftovers.
    s.execute(text("TRUNCATE TABLE claim_evidence, analysis_claims CASCADE"))
    s.commit()
    yield s
    s.execute(text("TRUNCATE TABLE claim_evidence, analysis_claims CASCADE"))
    s.commit()
    s.close()


def _paper(session, source_id: str = "p1") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _evidence(session, paper: Paper) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="limitations", section="Discussion", text="a limitation",
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _claim(
    session, evidence: Evidence, claim_type: str = "inference", status: str = "pending",
    source_table: str = "candidate_gaps",
) -> AnalysisClaim:
    claim = AnalysisClaim(
        claim_type=claim_type, claim_text="a claim", confidence="medium", status=status,
        source_table=source_table, source_id=uuid.uuid4(),
    )
    session.add(claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relationship="supports"))
    session.commit()
    return claim


def test_list_claims_returns_claims_with_evidence(client, session) -> None:
    paper = _paper(session)
    evidence = _evidence(session, paper)
    claim = _claim(session, evidence)

    body = client.get("/api/claims").json()

    assert body["total"] == 1
    item = body["items"][0]
    assert item["id"] == str(claim.id)
    assert len(item["evidence"]) == 1
    assert item["evidence"][0]["paper_title"] == "A Paper"
    assert item["evidence"][0]["relationship"] == "supports"


def test_list_claims_filters_by_status(client, session) -> None:
    paper = _paper(session)
    evidence = _evidence(session, paper)
    _claim(session, evidence, status="pending")
    _claim(session, evidence, status="approved")

    body = client.get("/api/claims", params={"status": "approved"}).json()

    assert body["total"] == 1
    assert body["items"][0]["status"] == "approved"


def test_list_claims_filters_by_claim_type(client, session) -> None:
    paper = _paper(session)
    evidence = _evidence(session, paper)
    _claim(session, evidence, claim_type="fact")
    _claim(session, evidence, claim_type="inference")

    body = client.get("/api/claims", params={"claim_type": "fact"}).json()

    assert body["total"] == 1
    assert body["items"][0]["claim_type"] == "fact"


def test_list_claims_filters_by_source_table(client, session) -> None:
    paper = _paper(session)
    evidence = _evidence(session, paper)
    _claim(session, evidence, source_table="candidate_gaps")
    _claim(session, evidence, source_table="research_assessments")

    body = client.get("/api/claims", params={"source_table": "research_assessments"}).json()

    assert body["total"] == 1
    assert body["items"][0]["source_table"] == "research_assessments"


def test_list_claims_rejects_invalid_status(client) -> None:
    assert client.get("/api/claims", params={"status": "bogus"}).status_code == 422


def test_list_claims_rejects_invalid_claim_type(client) -> None:
    assert client.get("/api/claims", params={"claim_type": "bogus"}).status_code == 422


def test_list_claims_rejects_invalid_source_table(client) -> None:
    assert client.get("/api/claims", params={"source_table": "bogus"}).status_code == 422


def test_list_claims_empty_when_none_exist(client, session) -> None:
    body = client.get("/api/claims").json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


# --- /api/extracted-claims: Sec 28 raw per-paper extraction claims
# (problem/method/.../applications), a different table (extracted_claims)
# from analysis_claims above - no status or source_table, since a paper's
# extraction has neither concept.


def _extracted_evidence(
    session, paper: Paper, claim_type: str = "problem", extraction_method: str = "hybrid",
) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section="introduction", text=f"a {claim_type} sentence",
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _extracted_claim(session, paper: Paper, evidence: Evidence, claim_type: str = "problem") -> ExtractedClaim:
    claim = ExtractedClaim(
        paper_id=paper.id, claim_type=claim_type, text=evidence.text, evidence_id=evidence.id, confidence="medium",
    )
    session.add(claim)
    session.commit()
    return claim


def test_list_extracted_claims_returns_claims_with_paper_context(client, session) -> None:
    paper = _paper(session)
    evidence = _extracted_evidence(session, paper, claim_type="applications")
    claim = _extracted_claim(session, paper, evidence, claim_type="applications")

    body = client.get("/api/extracted-claims").json()

    assert body["total"] == 1
    item = body["items"][0]
    assert item["id"] == str(claim.id)
    assert item["claim_type"] == "applications"
    assert item["text"] == evidence.text
    assert item["section"] == "introduction"
    assert item["extraction_method"] == "hybrid"
    assert item["paper_id"] == str(paper.id)
    assert item["paper_title"] == "A Paper"


def test_list_extracted_claims_filters_by_claim_type(client, session) -> None:
    paper = _paper(session)
    problem_evidence = _extracted_evidence(session, paper, claim_type="problem")
    _extracted_claim(session, paper, problem_evidence, claim_type="problem")
    apps_evidence = _extracted_evidence(session, paper, claim_type="applications")
    _extracted_claim(session, paper, apps_evidence, claim_type="applications")

    body = client.get("/api/extracted-claims", params={"claim_type": "applications"}).json()

    assert body["total"] == 1
    assert body["items"][0]["claim_type"] == "applications"


def test_list_extracted_claims_excludes_stub_rows(client, session) -> None:
    paper = _paper(session)
    stub_evidence = _extracted_evidence(session, paper, extraction_method="stub")
    _extracted_claim(session, paper, stub_evidence)

    body = client.get("/api/extracted-claims").json()

    assert body["total"] == 0


def test_list_extracted_claims_rejects_invalid_claim_type(client) -> None:
    assert client.get("/api/extracted-claims", params={"claim_type": "bogus"}).status_code == 422


def test_list_extracted_claims_empty_when_none_exist(client, session) -> None:
    body = client.get("/api/extracted-claims").json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}
