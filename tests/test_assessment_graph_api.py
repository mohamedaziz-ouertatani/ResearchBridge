from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper
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


@pytest.fixture(autouse=True)
def _disable_ollama_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # This file POSTs to /api/assessments, which now calls build_assessment
    # with enable_llm_stages=ollama_enabled() (2026-09-04) - see
    # test_assessment_api.py's identical fixture for why an explicit
    # "false" (not delenv) is required to be immune to create_app()'s own
    # load_dotenv(override=False) call, regardless of fixture ordering.
    monkeypatch.setenv("OLLAMA_ENABLED", "false")


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
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def test_graph_returns_input_and_paper_nodes(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    created = client.post(
        "/api/assessments", json={"raw_text": "graph transformers for fraud detection"}
    ).json()

    response = client.get(f"/api/assessments/{created['id']}/graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 2
    input_node = next(n for n in body["nodes"] if n["type"] == "input")
    paper_node = next(n for n in body["nodes"] if n["type"] == "paper")
    assert input_node["distance_to_input"] is None
    assert paper_node["distance_to_input"] == pytest.approx(0.0, abs=1e-6)
    assert len(body["edges"]) == 1
    assert body["edges"][0]["source"] == "input"


def test_graph_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/graph")

    assert response.status_code == 404


def test_graph_returns_only_input_node_for_empty_corpus(client, session) -> None:
    created = client.post("/api/assessments", json={"raw_text": "nothing in the corpus"}).json()

    response = client.get(f"/api/assessments/{created['id']}/graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 1
    assert body["edges"] == []
