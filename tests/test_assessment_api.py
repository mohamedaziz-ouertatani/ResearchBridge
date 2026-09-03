from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper, ResearchAssessment, ResearchInput
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def client(session_factory, embedder):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def _add_paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def _add_claim(session, paper: Paper, claim_type: str, text: str) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium"))


def _add_completed_assessment(
    session,
    raw_text: str,
    recommendation: str | None,
    novelty_level: str = "not_assessed",
    technical_feasibility_level: str = "not_assessed",
) -> ResearchAssessment:
    """Directly inserts a completed assessment with specific categorical
    fields, bypassing the real pipeline - used by sort/filter tests below,
    which exercise the list endpoint's own logic rather than recommendation
    computation (already covered by the pipeline tests above)."""
    research_input = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=raw_text)
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(
        id=uuid.uuid4(),
        research_input_id=research_input.id,
        status="completed",
        novelty_level=novelty_level,
        technical_feasibility_level=technical_feasibility_level,
        recommendation=recommendation,
    )
    session.add(assessment)
    session.commit()
    return assessment


def test_post_assessment_creates_input_and_runs_the_pipeline(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only in offline settings")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["status"] == "completed"
    assert str(paper.id) in body["retrieved_paper_ids"]
    assert "evaluated only in offline settings" in body["comparison_summary"]
    # dimension coverage now drives novelty (see assessment/novelty.py): a
    # single retrieved paper can only ever reach "weak_evidence" per
    # dimension (2+ distinct papers are required for "established"), so an
    # exact-title match with just one corroborating paper reads "high", not
    # "low" - see tests/test_assessment_build.py's
    # test_novelty_is_high_when_a_single_paper_cannot_corroborate_dimension_coverage
    # for the same behavior traced end-to-end at the build_assessment level.
    assert body["novelty_level"] == "high"
    assert "Dimension coverage:" in body["novelty_reasoning"]


def test_post_assessment_novelty_not_assessed_without_any_evidence(client, session) -> None:
    body = client.post("/api/assessments", json={"raw_text": "an idea with no related papers in the corpus"}).json()

    assert body["novelty_level"] == "not_assessed"
    assert body["novelty_reasoning"] is not None


def test_post_assessment_includes_research_gap_from_explicit_claim(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["research_gap_source"] == "input_specific"
    assert "no real-time evaluation exists" in body["research_gap_text"]
    assert body["candidate_gap_id"] is None


def test_post_assessment_includes_potential_applications(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "applications", "real-time payment fraud screening")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["potential_applications"][0]["application"] == "real-time payment fraud screening"


def test_post_assessment_potential_applications_null_without_evidence(client, session) -> None:
    body = client.post("/api/assessments", json={"raw_text": "an idea with no related papers in the corpus"}).json()

    assert body["potential_applications"] is None


def test_post_assessment_includes_technical_feasibility(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "method", "a graph attention mechanism")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["technical_feasibility_level"] == "medium"
    assert paper.title in body["technical_feasibility_reasoning"]


def test_post_assessment_includes_risks_and_limitations(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert "evaluated only on offline datasets" in body["risks_and_limitations"]
    assert paper.title in body["risks_and_limitations"]


def test_post_assessment_includes_external_validation_needed(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert "not assessed" in body["external_validation_needed"].lower()


def test_post_assessment_includes_recommendation_and_confidence(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "research_gap", "no real-time evaluation exists")
    _add_claim(session, paper, "method", "a graph attention mechanism")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["recommendation"] is not None
    assert body["confidence"] is not None


def test_post_assessment_potential_opportunities_stays_null(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "applications", "real-time payment fraud screening")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["potential_opportunities"] is None


def test_post_assessment_requires_raw_text(client) -> None:
    assert client.post("/api/assessments", json={"raw_text": ""}).status_code == 422


def test_get_assessment_returns_previously_created_one(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    fetched = client.get(f"/api/assessments/{created['id']}").json()

    assert fetched["id"] == created["id"]
    assert fetched["research_input"]["raw_text"] == "graph transformers for fraud detection"


def test_get_assessment_404s_for_unknown_id(client) -> None:
    assert client.get(f"/api/assessments/{uuid.uuid4()}").status_code == 404


# --- POST /{id}/opportunities: on-demand LLM synthesis (see
# docs/superpowers/specs/2026-09-03-opportunities-synthesis-design.md) ------


def _create_with_applications(client, session, embedder) -> dict:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "applications", "real-time payment fraud screening")
    session.commit()
    return client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()


def test_opportunities_404s_for_unknown_assessment(client) -> None:
    response = client.post(f"/api/assessments/{uuid.uuid4()}/opportunities")

    assert response.status_code == 404


def test_opportunities_422s_without_any_potential_applications(client, session) -> None:
    body = client.post("/api/assessments", json={"raw_text": "an idea with no related papers in the corpus"}).json()
    assert body["potential_applications"] is None

    response = client.post(f"/api/assessments/{body['id']}/opportunities")

    assert response.status_code == 422


def test_opportunities_503s_when_ollama_is_disabled(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # explicit unset, not just "don't set it": .env (loaded by create_app()
    # -> load_config() -> load_dotenv, see test_qa_api.py's identical
    # pattern) may have OLLAMA_ENABLED=true for local Ollama development,
    # which load_dotenv(override=False) leaves in os.environ for the rest
    # of this pytest process once any earlier test's create_app() call has
    # loaded it - relying on "just don't set it" is not reliably "disabled"
    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    body = _create_with_applications(client, session, embedder)

    response = client.post(f"/api/assessments/{body['id']}/opportunities")

    assert response.status_code == 503
    refetched = client.get(f"/api/assessments/{body['id']}").json()
    assert refetched["potential_opportunities"] is None


def test_opportunities_synthesizes_and_persists_when_available(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import researchbridge.api.assessment_routes as routes_module
    from researchbridge.assessment.opportunity_synthesis import SynthesisResult, SynthesizedOpportunity

    body = _create_with_applications(client, session, embedder)

    def _fake_synthesize(applications):
        return SynthesisResult(
            opportunities=[
                SynthesizedOpportunity(tier="direct", opportunity="fraud-scoring API", source_application_indices=[1]),
                SynthesizedOpportunity(tier="adjacent", opportunity="risk platform", source_application_indices=[1]),
                SynthesizedOpportunity(tier="speculative", opportunity="fraud network", source_application_indices=[1]),
            ]
        )

    monkeypatch.setattr(routes_module, "synthesize_opportunities", _fake_synthesize)

    response = client.post(f"/api/assessments/{body['id']}/opportunities")

    assert response.status_code == 200
    opportunities = response.json()["potential_opportunities"]
    assert [o["tier"] for o in opportunities] == ["direct", "adjacent", "speculative"]
    assert opportunities[0]["opportunity"] == "fraud-scoring API"
    assert opportunities[0]["source_applications"] == [
        {
            "application": "real-time payment fraud screening",
            "paper_id": body["potential_applications"][0]["paper_id"],
            "paper_title": "graph transformers for fraud detection",
        }
    ]


def test_opportunities_persist_across_a_fresh_fetch(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import researchbridge.api.assessment_routes as routes_module
    from researchbridge.assessment.opportunity_synthesis import SynthesisResult, SynthesizedOpportunity

    body = _create_with_applications(client, session, embedder)
    monkeypatch.setattr(
        routes_module,
        "synthesize_opportunities",
        lambda applications: SynthesisResult(
            opportunities=[
                SynthesizedOpportunity(tier="direct", opportunity="a", source_application_indices=[1]),
                SynthesizedOpportunity(tier="adjacent", opportunity="b", source_application_indices=[1]),
                SynthesizedOpportunity(tier="speculative", opportunity="c", source_application_indices=[1]),
            ]
        ),
    )
    client.post(f"/api/assessments/{body['id']}/opportunities")

    fetched = client.get(f"/api/assessments/{body['id']}").json()

    assert [o["tier"] for o in fetched["potential_opportunities"]] == ["direct", "adjacent", "speculative"]


def test_assessment_returns_the_evidence_backing_each_field(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    roles = {item["role"] for item in body["evidence"]}
    assert "comparison" in roles
    assert "risk" in roles
    backing = next(item for item in body["evidence"] if item["role"] == "risk")
    assert backing["text"] == "evaluated only on offline datasets"
    assert backing["paper_title"] == "graph transformers for fraud detection"
    assert backing["paper_id"] == str(paper.id)


def test_assessment_evidence_is_empty_when_nothing_was_grounded(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")  # no claims
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["evidence"] == []


def test_fetched_assessment_also_carries_its_evidence(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()
    fetched = client.get(f"/api/assessments/{created['id']}").json()

    assert len(fetched["evidence"]) == len(created["evidence"])
    assert {i["role"] for i in fetched["evidence"]} == {i["role"] for i in created["evidence"]}


def test_upload_creates_a_document_input_and_runs_the_pipeline(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    text = b"Detecting fraud in financial transactions remains a major challenge. We propose graph transformers."
    response = client.post(
        "/api/assessments/upload", files={"file": ("paper.txt", text, "text/plain")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["research_input"]["input_type"] == "document"
    assert str(paper.id) in body["retrieved_paper_ids"]


def test_upload_extracts_text_from_a_pdf_by_filename(client, session, monkeypatch) -> None:
    import researchbridge.api.assessment_routes as routes_module

    monkeypatch.setattr(routes_module, "extract_text", lambda pdf_bytes: "extracted pdf text")

    response = client.post(
        "/api/assessments/upload", files={"file": ("paper.pdf", b"%PDF-fake-bytes", "application/pdf")}
    )

    assert response.status_code == 200
    assert response.json()["research_input"]["raw_text"] == "extracted pdf text"


def test_upload_sets_matched_paper_id_when_filename_matches_a_corpus_arxiv_paper(client, session, embedder) -> None:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id="2501.00348", title="graph transformers for fraud detection",
        abstract="", raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()

    text = b"Detecting fraud in financial transactions remains a major challenge."
    response = client.post(
        "/api/assessments/upload", files={"file": ("2501.00348.txt", text, "text/plain")}
    )

    assert response.json()["research_input"]["matched_paper_id"] == str(paper.id)


def test_upload_matched_paper_id_is_null_without_a_filename_match(client, session, embedder) -> None:
    text = b"Detecting fraud in financial transactions remains a major challenge."
    response = client.post(
        "/api/assessments/upload", files={"file": ("my_notes.txt", text, "text/plain")}
    )

    assert response.json()["research_input"]["matched_paper_id"] is None


def test_upload_rejects_an_empty_file(client) -> None:
    response = client.post("/api/assessments/upload", files={"file": ("paper.txt", b"", "text/plain")})

    assert response.status_code == 422


def test_upload_rejects_a_pdf_with_no_extractable_text(client, monkeypatch) -> None:
    import researchbridge.api.assessment_routes as routes_module

    monkeypatch.setattr(routes_module, "extract_text", lambda pdf_bytes: "   ")

    response = client.post(
        "/api/assessments/upload", files={"file": ("paper.pdf", b"%PDF-fake-bytes", "application/pdf")}
    )

    assert response.status_code == 422


def test_upload_rejects_a_file_named_pdf_that_is_not_actually_a_valid_pdf(client) -> None:
    # Item 6 (upload endpoint security review): before this was fixed, a
    # real (unmocked) invalid-PDF-bytes upload crashed with an unhandled
    # pymupdf.FileDataError -> 500, instead of a clean validation error.
    response = client.post(
        "/api/assessments/upload", files={"file": ("paper.pdf", b"not a real pdf at all", "application/pdf")}
    )

    assert response.status_code == 422
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_rejects_a_file_over_the_size_limit(client) -> None:
    # Item 6: `await file.read()` previously had no cap at all - an upload
    # of arbitrary size was read fully into memory before any validation,
    # a trivial memory-exhaustion DoS.
    import researchbridge.api.assessment_routes as routes_module

    oversized = b"x" * (routes_module.MAX_UPLOAD_BYTES + 1)
    response = client.post("/api/assessments/upload", files={"file": ("paper.txt", oversized, "text/plain")})

    assert response.status_code == 413


def test_upload_accepts_a_file_right_at_the_size_limit(client, embedder) -> None:
    import researchbridge.api.assessment_routes as routes_module

    at_limit = b"Detecting fraud in financial transactions remains a major challenge. " * 1000
    at_limit = at_limit[: routes_module.MAX_UPLOAD_BYTES]
    response = client.post("/api/assessments/upload", files={"file": ("paper.txt", at_limit, "text/plain")})

    assert response.status_code == 200


def test_new_assessment_is_not_human_reviewed_by_default(client) -> None:
    body = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()

    assert body["human_reviewed"] is False


def test_review_marks_assessment_as_human_reviewed(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()

    response = client.put(f"/api/assessments/{created['id']}/review", json={"human_reviewed": True})

    assert response.status_code == 200
    assert response.json()["human_reviewed"] is True


def test_review_can_toggle_back_to_unreviewed(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()
    client.put(f"/api/assessments/{created['id']}/review", json={"human_reviewed": True})

    response = client.put(f"/api/assessments/{created['id']}/review", json={"human_reviewed": False})

    assert response.json()["human_reviewed"] is False


def test_review_404s_for_unknown_assessment(client) -> None:
    response = client.put(f"/api/assessments/{uuid.uuid4()}/review", json={"human_reviewed": True})

    assert response.status_code == 404


def test_exclusion_does_not_affect_an_existing_assessments_evidence(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()
    assert created["evidence"]  # sanity check: the assessment has real evidence before exclusion

    exclude_response = client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})
    assert exclude_response.status_code == 200

    fetched = client.get(f"/api/assessments/{created['id']}").json()

    assert fetched["evidence"] == created["evidence"]
    assert fetched["comparison_summary"] == created["comparison_summary"]
    assert paper.title in fetched["comparison_summary"]


def test_review_persists_across_a_fresh_fetch(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()
    client.put(f"/api/assessments/{created['id']}/review", json={"human_reviewed": True})

    fetched = client.get(f"/api/assessments/{created['id']}").json()

    assert fetched["human_reviewed"] is True


def test_rerun_creates_a_new_assessment_for_the_same_input(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    rerun = client.post(f"/api/assessments/{created['id']}/rerun")

    assert rerun.status_code == 200
    body = rerun.json()
    assert body["id"] != created["id"]
    assert body["research_input"]["id"] == created["research_input"]["id"]
    assert body["status"] == "completed"


def test_rerun_picks_up_newly_ingested_evidence(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()
    # p1 was retrieved (relevant) but had no application claim yet - "no
    # evidence" (empty list), distinct from "no relevant papers at all"
    # (None) - see assessment/applications.py's status field
    assert created["potential_applications"] == []

    # same title as the query text so the (hash-based) fake embedder places it
    # at distance 0.0, well within assess_applications' relevance gate
    paper2 = _add_paper(session, embedder, "p2", "graph transformers for fraud detection")
    _add_claim(session, paper2, "applications", "real-time payment fraud screening")
    session.commit()

    rerun = client.post(f"/api/assessments/{created['id']}/rerun").json()

    assert rerun["potential_applications"][0]["application"] == "real-time payment fraud screening"


def test_rerun_404s_for_unknown_assessment(client) -> None:
    response = client.post(f"/api/assessments/{uuid.uuid4()}/rerun")

    assert response.status_code == 404


def test_delete_removes_the_assessment(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea"}).json()

    response = client.delete(f"/api/assessments/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"/api/assessments/{created['id']}").status_code == 404


def test_delete_removes_every_rerun_in_the_same_thread(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()
    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()
    rerun = client.post(f"/api/assessments/{created['id']}/rerun").json()

    # Deleting the ORIGINAL id must also remove the rerun - it's the same
    # logical entry from the list page's point of view (latest-per-input).
    response = client.delete(f"/api/assessments/{created['id']}")

    assert response.status_code == 204
    assert client.get(f"/api/assessments/{created['id']}").status_code == 404
    assert client.get(f"/api/assessments/{rerun['id']}").status_code == 404


def test_delete_removes_the_assessment_from_the_list(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea"}).json()

    client.delete(f"/api/assessments/{created['id']}")

    body = client.get("/api/assessments", params={"review": "all"}).json()
    assert created["id"] not in [item["id"] for item in body["items"]]


def test_delete_404s_for_unknown_assessment(client) -> None:
    response = client.delete(f"/api/assessments/{uuid.uuid4()}")

    assert response.status_code == 404


def test_history_lists_all_assessments_for_the_same_input_newest_first(client, session) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()
    rerun = client.post(f"/api/assessments/{created['id']}/rerun").json()

    history = client.get(f"/api/assessments/{created['id']}/history").json()

    assert [item["id"] for item in history] == [rerun["id"], created["id"]]


def test_history_items_carry_enough_to_distinguish_entries(client, session) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea with no related papers"}).json()

    history = client.get(f"/api/assessments/{created['id']}/history").json()

    item = history[0]
    assert item["id"] == created["id"]
    assert item["status"] == "completed"
    assert item["novelty_level"] == "not_assessed"
    assert item["human_reviewed"] is False
    assert "created_at" in item


def test_history_404s_for_unknown_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/history")

    assert response.status_code == 404


def test_export_docx_returns_a_docx_file(client, session, embedder) -> None:
    import docx

    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    response = client.get(f"/api/assessments/{created['id']}/export.docx")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert f"assessment-{created['id']}.docx" in response.headers["content-disposition"]
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(response.content)).paragraphs)
    assert "evaluated only on offline datasets" in text


def test_export_docx_404s_for_unknown_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/export.docx")

    assert response.status_code == 404


def test_export_pdf_returns_a_pdf_file(client, session, embedder) -> None:
    import pymupdf

    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    created = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    response = client.get(f"/api/assessments/{created['id']}/export.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert f"assessment-{created['id']}.pdf" in response.headers["content-disposition"]
    with pymupdf.open(stream=response.content, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    assert "evaluated only on offline datasets" in text


def test_export_pdf_404s_for_unknown_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/export.pdf")

    assert response.status_code == 404


def test_list_assessments_returns_newest_first(client) -> None:
    first = client.post("/api/assessments", json={"raw_text": "idea one"}).json()
    second = client.post("/api/assessments", json={"raw_text": "idea two"}).json()

    body = client.get("/api/assessments").json()

    assert [item["id"] for item in body["items"]] == [second["id"], first["id"]]
    assert body["total"] == 2


def test_list_assessments_collapses_rerun_history_to_latest(client) -> None:
    created = client.post("/api/assessments", json={"raw_text": "an idea"}).json()
    rerun = client.post(f"/api/assessments/{created['id']}/rerun").json()

    body = client.get("/api/assessments").json()

    assert body["total"] == 1
    assert body["items"][0]["id"] == rerun["id"]


def test_list_assessments_includes_input_preview(client) -> None:
    client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    item = client.get("/api/assessments").json()["items"][0]

    assert "graph transformers for fraud detection" in item["input_preview"]
    assert item["input_type"] == "idea"


def test_list_assessments_filters_needs_review(client) -> None:
    unreviewed = client.post("/api/assessments", json={"raw_text": "idea one"}).json()
    reviewed = client.post("/api/assessments", json={"raw_text": "idea two"}).json()
    client.put(f"/api/assessments/{reviewed['id']}/review", json={"human_reviewed": True})

    body = client.get("/api/assessments", params={"review": "needs_review"}).json()

    assert [item["id"] for item in body["items"]] == [unreviewed["id"]]


def test_list_assessments_filters_reviewed(client) -> None:
    client.post("/api/assessments", json={"raw_text": "idea one"}).json()
    reviewed = client.post("/api/assessments", json={"raw_text": "idea two"}).json()
    client.put(f"/api/assessments/{reviewed['id']}/review", json={"human_reviewed": True})

    body = client.get("/api/assessments", params={"review": "reviewed"}).json()

    assert [item["id"] for item in body["items"]] == [reviewed["id"]]


def test_list_assessments_rejects_invalid_review_filter(client) -> None:
    assert client.get("/api/assessments", params={"review": "bogus"}).status_code == 422


def test_list_assessments_paginates(client) -> None:
    for i in range(3):
        client.post("/api/assessments", json={"raw_text": f"idea {i}"})

    body = client.get("/api/assessments", params={"limit": 2, "offset": 1}).json()

    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_assessments_includes_technical_feasibility_level(client, session) -> None:
    _add_completed_assessment(session, "idea", "MEDIUM PRIORITY", technical_feasibility_level="high")

    item = client.get("/api/assessments").json()["items"][0]

    assert item["technical_feasibility_level"] == "high"


def test_list_assessments_sort_priority_orders_by_recommendation_tier(client, session) -> None:
    insufficient = _add_completed_assessment(session, "insufficient", "INSUFFICIENT EVIDENCE")
    high = _add_completed_assessment(session, "high", "HIGH PRIORITY")
    medium = _add_completed_assessment(session, "medium", "MEDIUM PRIORITY")
    low = _add_completed_assessment(session, "low", "LOW PRIORITY")
    needs_review = _add_completed_assessment(session, "needs review", "REQUIRES HUMAN REVIEW")
    unassessed = _add_completed_assessment(session, "unassessed", None)

    body = client.get("/api/assessments", params={"sort": "priority"}).json()

    assert [item["id"] for item in body["items"]] == [
        str(high.id),
        str(medium.id),
        str(low.id),
        str(needs_review.id),
        str(insufficient.id),
        str(unassessed.id),
    ]


def test_list_assessments_default_sort_is_newest_not_priority(client, session) -> None:
    _add_completed_assessment(session, "first", "HIGH PRIORITY")
    second = _add_completed_assessment(session, "second", "INSUFFICIENT EVIDENCE")

    body = client.get("/api/assessments").json()

    assert body["items"][0]["id"] == str(second.id)  # newest first, not priority-ordered


def test_list_assessments_rejects_invalid_sort(client) -> None:
    assert client.get("/api/assessments", params={"sort": "bogus"}).status_code == 422


def test_list_assessments_filters_by_novelty_level(client, session) -> None:
    high_novelty = _add_completed_assessment(session, "high novelty", "HIGH PRIORITY", novelty_level="high")
    _add_completed_assessment(session, "low novelty", "LOW PRIORITY", novelty_level="low")

    body = client.get("/api/assessments", params={"novelty": "high"}).json()

    assert [item["id"] for item in body["items"]] == [str(high_novelty.id)]


def test_list_assessments_filters_by_technical_feasibility_level(client, session) -> None:
    high_feasibility = _add_completed_assessment(
        session, "high feasibility", "HIGH PRIORITY", technical_feasibility_level="high"
    )
    _add_completed_assessment(session, "low feasibility", "LOW PRIORITY", technical_feasibility_level="low")

    body = client.get("/api/assessments", params={"feasibility": "high"}).json()

    assert [item["id"] for item in body["items"]] == [str(high_feasibility.id)]
