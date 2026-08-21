from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import (
    EMBEDDING_DIM,
    Embedding,
    EmbeddingRun,
    Evidence,
    ExtractedClaim,
    ExtractionRun,
    IngestionRun,
    Paper,
)
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


def _add_paper(session, embedder, source_id: str, embed: bool = False, claim: bool = False) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=f"paper {source_id}", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    if embed:
        [vector] = embedder.embed_texts([paper.title])
        session.add(
            Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
        )
    if claim:
        evidence = Evidence(
            paper_id=paper.id, evidence_type="method", section=None, text="a method",
            extraction_method="hybrid", model_version="v1", confidence="medium",
        )
        session.add(evidence)
        session.flush()
        session.add(
            ExtractedClaim(paper_id=paper.id, claim_type="method", text="a method", evidence_id=evidence.id, confidence="medium")
        )
    return paper


def test_pipeline_status_reports_corpus_coverage(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", embed=True, claim=True)
    _add_paper(session, embedder, "p2", embed=True, claim=False)
    _add_paper(session, embedder, "p3", embed=False, claim=False)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["total_papers"] == 3
    assert body["papers_with_embeddings"] == 2
    assert body["papers_with_claims"] == 1


def test_pipeline_status_lists_recent_ingestion_runs(client, session) -> None:
    session.add(
        IngestionRun(
            source="arxiv", status="completed", started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc), records_fetched=100, records_inserted=90,
            records_duplicate=8, records_failed=2,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert len(body["ingestion_runs"]) == 1
    run = body["ingestion_runs"][0]
    assert run["status"] == "completed"
    assert run["counts"]["records_fetched"] == 100
    assert run["counts"]["records_inserted"] == 90
    assert run["counts"]["records_duplicate"] == 8
    assert run["counts"]["records_failed"] == 2


def test_pipeline_status_lists_recent_extraction_runs(client, session) -> None:
    session.add(
        ExtractionRun(
            extractor_name="hybrid", status="completed", started_at=datetime.now(timezone.utc),
            papers_processed=50, claims_created=120, candidates_rejected=5,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert len(body["extraction_runs"]) == 1
    run = body["extraction_runs"][0]
    assert run["counts"]["papers_processed"] == 50
    assert run["counts"]["claims_created"] == 120
    assert run["counts"]["candidates_rejected"] == 5


def test_pipeline_status_lists_recent_embedding_runs(client, session) -> None:
    session.add(
        EmbeddingRun(
            model_name="fake-embedder-v1", status="completed", started_at=datetime.now(timezone.utc),
            papers_processed=200, papers_skipped=3,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert len(body["embedding_runs"]) == 1
    run = body["embedding_runs"][0]
    assert run["counts"]["papers_processed"] == 200
    assert run["counts"]["papers_skipped"] == 3


def test_pipeline_status_surfaces_error_summary(client, session) -> None:
    session.add(
        IngestionRun(
            source="arxiv", status="failed", started_at=datetime.now(timezone.utc),
            records_fetched=0, records_inserted=0, records_duplicate=0, records_failed=0,
            error_summary="connection timed out",
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["ingestion_runs"][0]["error_summary"] == "connection timed out"


def test_exclude_sets_excluded_at(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()

    response = client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    assert response.status_code == 200
    assert response.json()["excluded_at"] is not None


def test_exclude_false_clears_excluded_at(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()
    client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    response = client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": False})

    assert response.json()["excluded_at"] is None


def test_exclude_404s_for_unknown_paper(client) -> None:
    response = client.put(f"/api/admin/papers/{uuid.uuid4()}/exclude", json={"excluded": True})

    assert response.status_code == 404


def test_get_paper_still_works_after_exclusion(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()
    client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    response = client.get(f"/api/papers/{paper.id}")

    assert response.status_code == 200
    assert response.json()["excluded_at"] is not None


def test_pipeline_status_orders_runs_most_recent_first(client, session) -> None:
    now = datetime.now(timezone.utc)
    session.add(
        IngestionRun(
            source="arxiv", status="completed", started_at=now - timedelta(days=1),
            records_fetched=1, records_inserted=1, records_duplicate=0, records_failed=0,
        )
    )
    session.add(
        IngestionRun(
            source="arxiv", status="completed", started_at=now,
            records_fetched=2, records_inserted=2, records_duplicate=0, records_failed=0,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert [run["counts"]["records_fetched"] for run in body["ingestion_runs"]] == [2, 1]


def test_pipeline_status_ingestion_runs_include_source(client, session) -> None:
    session.add(
        IngestionRun(
            source="springer", status="completed", started_at=datetime.now(timezone.utc),
            records_fetched=5, records_inserted=5, records_duplicate=0, records_failed=0,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["ingestion_runs"][0]["source"] == "springer"


def test_pipeline_status_extraction_and_embedding_runs_have_no_source(client, session) -> None:
    session.add(
        ExtractionRun(
            extractor_name="hybrid", status="completed", started_at=datetime.now(timezone.utc),
            papers_processed=1, claims_created=1, candidates_rejected=0,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["extraction_runs"][0]["source"] is None


def test_pipeline_status_reports_nothing_running_by_default(client) -> None:
    body = client.get("/api/admin/pipeline").json()

    assert body["running"] == {
        "ingestion_arxiv": False,
        "ingestion_springer": False,
        "extraction": False,
        "embedding": False,
    }


def test_trigger_arxiv_ingestion_calls_trigger_with_no_extra_flags_by_default(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    response = client.post("/api/admin/ingestion/arxiv/run", json={})

    assert response.status_code == 200
    assert response.json() == {"started": True, "pipeline": "ingestion_arxiv", "log_file": "x.log"}
    assert calls == [("ingestion_arxiv", "researchbridge.ingestion.cli", [])]


def test_trigger_arxiv_ingestion_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post(
        "/api/admin/ingestion/arxiv/run",
        json={"search_query": "cat:cs.CL", "page_size": 50, "max_pages": 2},
    )

    assert calls == [
        (
            "ingestion_arxiv",
            "researchbridge.ingestion.cli",
            ["--search-query", "cat:cs.CL", "--page-size", "50", "--max-pages", "2"],
        )
    ]


def test_trigger_springer_ingestion_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post(
        "/api/admin/ingestion/springer/run",
        json={"query": '"deep learning"', "page_size": 25, "max_pages": 4},
    )

    assert calls == [
        (
            "ingestion_springer",
            "researchbridge.ingestion.cli_springer",
            ["--query", '"deep learning"', "--page-size", "25", "--max-pages", "4"],
        )
    ]


def test_trigger_extraction_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/extraction/run", json={"limit": 10, "extractor": "heuristic"})

    assert calls == [
        (
            "extraction",
            "researchbridge.extraction.cli",
            ["--limit", "10", "--extractor", "heuristic"],
        )
    ]


def test_trigger_embedding_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/embedding/run", json={"limit": 25})

    assert calls == [("embedding", "researchbridge.embedding.cli_embed", ["--limit", "25"])]


def test_trigger_409s_when_already_running(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module
    from researchbridge.api.pipeline_triggers import PipelineAlreadyRunning

    def _raise(key, module, args):
        raise PipelineAlreadyRunning(key)

    monkeypatch.setattr(routes_module, "trigger", _raise)

    response = client.post("/api/admin/embedding/run", json={})

    assert response.status_code == 409


def test_pipeline_status_reflects_is_running(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "is_running", lambda key: key == "extraction")

    body = client.get("/api/admin/pipeline").json()

    assert body["running"] == {
        "ingestion_arxiv": False,
        "ingestion_springer": False,
        "extraction": True,
        "embedding": False,
    }
