from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper
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


def test_post_assessment_creates_input_and_runs_the_pipeline(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only in offline settings")
    session.commit()

    body = client.post("/api/assessments", json={"raw_text": "graph transformers for fraud detection"}).json()

    assert body["status"] == "completed"
    assert str(paper.id) in body["retrieved_paper_ids"]
    assert "evaluated only in offline settings" in body["comparison_summary"]
    assert body["novelty_level"] == "low"  # identical text to the paper title -> distance 0.0
    assert paper.title in body["novelty_reasoning"]


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
