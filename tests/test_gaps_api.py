from __future__ import annotations

import uuid
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import CandidateGap, CandidateGapEvidence, Evidence, ExtractedClaim, Paper


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
    yield s
    # candidate_gaps/candidate_gap_evidence aren't in conftest's TRUNCATE list
    s.execute(text("TRUNCATE TABLE candidate_gap_evidence, candidate_gaps CASCADE"))
    s.commit()
    s.close()


def _add_paper(session, source_id: str, title: str = "A Paper") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=title, abstract="Abstract.",
        publication_date=date(2024, 1, 1), url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_gap(
    session,
    seed: Paper,
    contributing: Paper,
    observation: str = "Recurring pattern across 3 related papers: \"tested only offline\"",
    status: str = "pending",
) -> CandidateGap:
    evidence = Evidence(
        paper_id=contributing.id, evidence_type="limitations", section=None, text="tested only offline",
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(
            paper_id=contributing.id, claim_type="limitations", text="tested only offline",
            evidence_id=evidence.id, confidence="medium",
        )
    )
    session.flush()

    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed.id, observation=observation, gap_type="inference",
        status=status, contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    session.add(CandidateGapEvidence(candidate_gap_id=gap.id, evidence_id=evidence.id))
    session.commit()
    return gap


def test_list_gaps_defaults_to_pending(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    _add_gap(session, seed, other, status="pending")
    _add_gap(session, seed, other, status="approved")

    body = client.get("/api/gaps").json()

    assert body["total"] == 1
    assert body["items"][0]["status"] == "pending"


def test_list_gaps_includes_seed_paper_title_and_evidence(client, session) -> None:
    seed = _add_paper(session, "seed", title="Seed Paper Title")
    other = _add_paper(session, "other", title="Contributing Paper Title")
    _add_gap(session, seed, other)

    item = client.get("/api/gaps").json()["items"][0]

    assert item["seed_paper_title"] == "Seed Paper Title"
    assert len(item["evidence"]) == 1
    assert item["evidence"][0]["paper_title"] == "Contributing Paper Title"
    assert item["evidence"][0]["text"] == "tested only offline"


def test_list_gaps_filters_by_status_all(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    _add_gap(session, seed, other, status="pending")
    _add_gap(session, seed, other, status="rejected")

    body = client.get("/api/gaps", params={"status": "all"}).json()

    assert body["total"] == 2


def test_list_gaps_rejects_invalid_status(client) -> None:
    assert client.get("/api/gaps", params={"status": "bogus"}).status_code == 422


def test_review_gap_approves_and_sets_reviewed_at(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    body = client.put(f"/api/gaps/{gap.id}", json={"status": "approved", "review_note": "looks solid"}).json()

    assert body["status"] == "approved"
    assert body["review_note"] == "looks solid"
    session.expire_all()
    refreshed = session.get(CandidateGap, gap.id)
    assert refreshed.reviewed_at is not None


def test_review_gap_rejects_with_note(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    body = client.put(f"/api/gaps/{gap.id}", json={"status": "rejected", "review_note": "too weak"}).json()

    assert body["status"] == "rejected"
    assert body["review_note"] == "too weak"


def test_review_gap_saves_ratings(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    body = client.put(
        f"/api/gaps/{gap.id}",
        json={
            "status": "approved",
            "correctness_rating": 3,
            "relevance_rating": 2,
            "novelty_rating": 3,
            "evidence_support_rating": 1,
            "usefulness_rating": 2,
        },
    ).json()

    assert body["correctness_rating"] == 3
    assert body["relevance_rating"] == 2
    assert body["novelty_rating"] == 3
    assert body["evidence_support_rating"] == 1
    assert body["usefulness_rating"] == 2


def test_review_gap_ratings_default_to_null(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    body = client.put(f"/api/gaps/{gap.id}", json={"status": "approved"}).json()

    assert body["correctness_rating"] is None
    assert body["usefulness_rating"] is None


def test_review_gap_rejects_out_of_range_rating(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    response = client.put(f"/api/gaps/{gap.id}", json={"status": "approved", "correctness_rating": 5})

    assert response.status_code == 422


def test_review_gap_rejects_invalid_status(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)

    response = client.put(f"/api/gaps/{gap.id}", json={"status": "bogus"})

    assert response.status_code == 422


def test_review_gap_404s_for_unknown_id(client) -> None:
    response = client.put(f"/api/gaps/{uuid.uuid4()}", json={"status": "approved"})
    assert response.status_code == 404


def test_gap_response_includes_status_and_resolution_note(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)
    gap.gap_status = "strong_gap"
    gap.resolution_note = "note text"
    session.commit()

    body = client.get("/api/gaps", params={"status": "pending"}).json()
    item = body["items"][0]

    assert item["gap_status"] == "strong_gap"
    assert item["resolution_note"] == "note text"


def test_gap_evidence_includes_claim_type_and_role(client, session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")
    gap = _add_gap(session, seed, other)
    link = session.execute(
        text("SELECT id, evidence_id FROM candidate_gap_evidence WHERE candidate_gap_id = :gap_id"),
        {"gap_id": gap.id},
    ).one()
    session.execute(
        text(
            "UPDATE candidate_gap_evidence "
            "SET claim_role = :claim_role, self_resolution_signal = :self_resolution_signal, "
            "field_scope_signal = :field_scope_signal, own_contribution_overlap = :own_contribution_overlap "
            "WHERE id = :id"
        ),
        {
            "claim_role": "anchor",
            "self_resolution_signal": False,
            "field_scope_signal": True,
            "own_contribution_overlap": 0.2,
            "id": link.id,
        },
    )
    session.execute(
        text(
            "UPDATE extracted_claims SET claim_type = :claim_type, validation_tier = :validation_tier "
            "WHERE evidence_id = :evidence_id"
        ),
        {"claim_type": "research_gap", "validation_tier": "strong", "evidence_id": link.evidence_id},
    )
    # claim_type in the API response is sourced from Evidence.evidence_type
    # (see serializers.py::_evidence_by_gap), NOT ExtractedClaim.claim_type -
    # _add_gap hardcodes evidence_type="limitations", so it must be updated
    # here too for the response to actually come back as "research_gap".
    session.execute(
        text("UPDATE evidence SET evidence_type = :evidence_type WHERE id = :evidence_id"),
        {"evidence_type": "research_gap", "evidence_id": link.evidence_id},
    )
    session.commit()

    body = client.get("/api/gaps", params={"status": "pending"}).json()
    evidence = body["items"][0]["evidence"][0]

    assert evidence["claim_type"] == "research_gap"
    assert evidence["claim_role"] == "anchor"
    assert evidence["validation_tier"] == "strong"
    assert evidence["self_resolution_signal"] is False
    assert evidence["field_scope_signal"] is True
    assert evidence["own_contribution_overlap"] == 0.2


def test_db_rejects_invalid_status_bypassing_the_api(session) -> None:
    seed = _add_paper(session, "seed")
    other = _add_paper(session, "other")

    with pytest.raises(Exception, match="ck_candidate_gaps_status"):
        _add_gap(session, seed, other, status="bogus")
    session.rollback()  # the failed commit leaves the session unusable until rolled back
