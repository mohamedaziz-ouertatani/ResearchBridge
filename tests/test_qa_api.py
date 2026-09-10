from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper, PaperFullTextChunk, QaQuestion
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


def _add_claim(session, paper: Paper, claim_type: str, text: str) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )


def test_ask_returns_ranked_quotes(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["hits"]) == 1
    hit = body["hits"][0]
    assert hit["text"] == "evaluated only on offline datasets"
    assert hit["paper_title"] == "graph transformers for fraud detection"
    assert hit["claim_type"] == "limitations"
    assert hit["paper_id"] == str(paper.id)
    assert isinstance(hit["score"], float)


def _add_chunk(session, embedder, paper, text, section="introduction", index=0) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        PaperFullTextChunk(
            paper_id=paper.id, section=section, paragraph_index=index, text=text,
            model_name=embedder.model_name, embedding=vector,
        )
    )


def test_ask_tags_fulltext_passages_with_source_passage(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_chunk(session, embedder, paper, "the exact question text")
    session.commit()

    response = client.post("/api/ask", json={"question": "the exact question text"})

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["source"] == "passage"
    assert hit["evidence_id"] is None


def test_ask_tags_claims_with_source_claim(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["source"] == "claim"


def test_ask_returns_empty_hits_when_no_evidence_exists(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_ask_returns_empty_hits_for_empty_corpus(client) -> None:
    response = client.post("/api/ask", json={"question": "anything at all"})

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_ask_rejects_empty_question(client) -> None:
    response = client.post("/api/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_rejects_missing_question(client) -> None:
    response = client.post("/api/ask", json={})

    assert response.status_code == 422


def test_ask_rejects_whitespace_only_question(client) -> None:
    response = client.post("/api/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_reports_summarization_available_true_when_enabled(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.json()["summarization_available"] is True


def test_ask_reports_summarization_available_true_by_default(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    response = client.post("/api/ask", json={"question": "anything"})

    assert response.json()["summarization_available"] is True


def test_ask_reports_summarization_available_false_when_disabled(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    response = client.post("/api/ask", json={"question": "anything"})

    assert response.json()["summarization_available"] is False


def test_summarize_returns_summary_when_enabled(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": "Grounded summary [1]."}}
    mock_response.raise_for_status = Mock()
    monkeypatch.setattr(
        "researchbridge.qa.summarize.requests.post", Mock(return_value=mock_response)
    )
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Grounded summary [1]."
    assert body["citations"] == [1]


def test_summarize_returns_503_when_disabled(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 503


def test_summarize_returns_503_when_ollama_unreachable(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    monkeypatch.setattr(
        "researchbridge.qa.summarize.requests.post",
        Mock(side_effect=requests.ConnectionError("refused")),
    )
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 503


def test_summarize_rejects_empty_hits(client) -> None:
    response = client.post("/api/ask/summarize", json={"question": "a question", "hits": []})

    assert response.status_code == 422


def test_collection_lifecycle_and_question_count(client) -> None:
    created = client.post("/api/qa/collections", json={"title": "Fraud reading"})
    assert created.status_code == 201
    collection_id = created.json()["id"]
    assert created.json()["question_count"] == 0

    listed = client.get("/api/qa/collections")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == collection_id
    assert listed.json()[0]["question_count"] == 0

    renamed = client.patch(f"/api/qa/collections/{collection_id}", json={"title": "Fraud notes"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Fraud notes"


def test_saved_question_preserves_grounded_hit_snapshot(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    collection_id = client.post("/api/qa/collections", json={"title": "Reading list"}).json()["id"]
    response = client.post(
        f"/api/qa/collections/{collection_id}/questions",
        json={"question": "graph transformers for fraud detection"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["collection_id"] == collection_id
    assert body["hits"][0]["text"] == "evaluated only on offline datasets"
    assert body["summary"] is None

    detail = client.get(f"/api/qa/collections/{collection_id}")
    assert detail.status_code == 200
    assert len(detail.json()["questions"]) == 1
    assert detail.json()["questions"][0]["hits"] == body["hits"]


def test_saved_question_summary_uses_persisted_hits(client, session, embedder, monkeypatch) -> None:
    paper = _add_paper(session, embedder, "p1", "a question paper")
    _add_claim(session, paper, "method", "a grounded quote")
    session.commit()
    collection_id = client.post("/api/qa/collections", json={"title": "Summaries"}).json()["id"]
    question = client.post(
        f"/api/qa/collections/{collection_id}/questions", json={"question": "a question paper"}
    ).json()

    def fake_summary(question_text, hits):
        assert question_text == "a question paper"
        assert len(hits) == 1
        assert hits[0].text == "a grounded quote"
        from researchbridge.qa.summarize import SummaryResult

        return SummaryResult(summary="Saved summary [1].", citations=[1])

    monkeypatch.setattr("researchbridge.api.qa_routes.summarize_quotes", fake_summary)
    response = client.post(f"/api/qa/questions/{question['id']}/summarize")

    assert response.status_code == 200
    assert response.json()["summary"] == "Saved summary [1]."
    assert response.json()["summary_citations"] == [1]


def test_deleting_collection_cascades_questions(client, session) -> None:
    collection_id = client.post("/api/qa/collections", json={"title": "To delete"}).json()["id"]
    question = client.post(
        f"/api/qa/collections/{collection_id}/questions", json={"question": "a question"}
    ).json()

    response = client.delete(f"/api/qa/collections/{collection_id}")
    assert response.status_code == 204
    assert client.get(f"/api/qa/collections/{collection_id}").status_code == 404
    assert session.get(QaQuestion, question["id"]) is None
