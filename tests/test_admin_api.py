from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import (
    EMBEDDING_DIM,
    CandidateGap,
    CitationFetchRun,
    Embedding,
    EmbeddingRun,
    Evidence,
    ExtractedClaim,
    ExtractionRun,
    IngestionError,
    IngestionRun,
    Paper,
    PaperCitation,
    ResearchAssessment,
    ResearchInput,
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
    # citation_fetch_runs has no FK to papers, so conftest's per-test
    # TRUNCATE ... CASCADE (which clears candidate_gaps for free via its
    # seed_paper_id FK) never reaches it - clear it explicitly before AND
    # after, since test order across files isn't guaranteed.
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    yield s
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    s.close()


def _add_paper(
    session, embedder, source_id: str, embed: bool = False, claim: bool = False, source: str = "arxiv",
    doi: str | None = None, excluded: bool = False,
) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title=f"paper {source_id}", abstract="",
        raw_metadata={}, ingestion_metadata={}, doi=doi,
        excluded_at=datetime.now(timezone.utc) if excluded else None,
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


def test_pipeline_status_reports_corpus_health_missing_doi(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", doi="10.1/abc")
    _add_paper(session, embedder, "p2", doi=None)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["corpus_health"]["missing_doi"] == 1


def test_pipeline_status_reports_corpus_health_excluded(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", excluded=True)
    _add_paper(session, embedder, "p2", excluded=False)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["corpus_health"]["excluded"] == 1


def test_pipeline_status_reports_corpus_health_claims_without_embeddings(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", claim=True, embed=False)
    _add_paper(session, embedder, "p2", claim=True, embed=True)
    _add_paper(session, embedder, "p3", claim=False, embed=False)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["corpus_health"]["claims_without_embeddings"] == 1


def test_pipeline_status_reports_corpus_health_no_citation_coverage(client, session, embedder) -> None:
    covered = _add_paper(session, embedder, "p1", doi="10.1/covered")
    _add_paper(session, embedder, "p2", doi="10.1/uncovered")
    uncovered_s2 = _add_paper(session, embedder, "p3", source="semantic_scholar")
    # a paper with no DOI and not semantic_scholar-sourced isn't eligible for
    # either citation source, so it shouldn't count as "no coverage"
    _add_paper(session, embedder, "p4", source="core", doi=None)
    session.flush()
    session.add(
        PaperCitation(
            citing_paper_id=covered.id, cited_paper_id=covered.id, source="crossref", confidence="high",
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["corpus_health"]["no_citation_coverage"] == 2  # p2 and p3
    assert uncovered_s2.id  # sanity: fixture used


def _add_gap(
    session, embedder, status: str = "pending", correctness_rating: int | None = None,
    relevance_rating: int | None = None, novelty_rating: int | None = None,
    evidence_support_rating: int | None = None, usefulness_rating: int | None = None,
) -> CandidateGap:
    paper = _add_paper(session, embedder, f"gap-seed-{uuid.uuid4()}")
    session.flush()
    gap = CandidateGap(
        seed_paper_id=paper.id, observation="a pattern", contributing_paper_count=3,
        similarity_threshold=0.8, detection_method="embedding_cosine", status=status,
        correctness_rating=correctness_rating, relevance_rating=relevance_rating,
        novelty_rating=novelty_rating, evidence_support_rating=evidence_support_rating,
        usefulness_rating=usefulness_rating,
    )
    session.add(gap)
    session.flush()
    return gap


def test_pipeline_status_reports_gap_stats(client, session, embedder) -> None:
    _add_gap(session, embedder, status="pending")
    _add_gap(session, embedder, status="pending")
    _add_gap(session, embedder, status="approved")
    _add_gap(session, embedder, status="rejected")
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["gap_stats"]["pending"] == 2
    assert body["gap_stats"]["approved"] == 1
    assert body["gap_stats"]["rejected"] == 1


def test_pipeline_status_gap_stats_zero_when_no_gaps(client) -> None:
    body = client.get("/api/admin/pipeline").json()

    assert body["gap_stats"] == {
        "pending": 0, "approved": 0, "rejected": 0,
        "mean_correctness": None, "mean_relevance": None, "mean_novelty": None,
        "mean_evidence_support": None, "mean_usefulness": None,
    }


def test_pipeline_status_reports_mean_gap_ratings(client, session, embedder) -> None:
    _add_gap(session, embedder, status="approved", correctness_rating=3, relevance_rating=2)
    _add_gap(session, embedder, status="approved", correctness_rating=1, relevance_rating=2)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["gap_stats"]["mean_correctness"] == 2.0
    assert body["gap_stats"]["mean_relevance"] == 2.0
    assert body["gap_stats"]["mean_novelty"] is None


def test_pipeline_status_reports_ingestion_errors_by_type(client, session) -> None:
    run = IngestionRun(
        source="arxiv", status="completed", started_at=datetime.now(timezone.utc),
        records_fetched=3, records_inserted=1, records_duplicate=0, records_failed=2,
    )
    session.add(run)
    session.flush()
    session.add(
        IngestionError(
            ingestion_run_id=run.id, source="arxiv", raw_payload={}, error_type="validation",
            error_detail="missing title",
        )
    )
    session.add(
        IngestionError(
            ingestion_run_id=run.id, source="arxiv", raw_payload={}, error_type="validation",
            error_detail="missing abstract",
        )
    )
    session.add(
        IngestionError(
            ingestion_run_id=run.id, source="arxiv", raw_payload={}, error_type="network",
            error_detail="timeout",
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["ingestion_errors_by_type"] == {"validation": 2, "network": 1}


def test_pipeline_status_ingestion_errors_by_type_empty_when_none(client) -> None:
    body = client.get("/api/admin/pipeline").json()

    assert body["ingestion_errors_by_type"] == {}


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


def test_pipeline_status_lists_recent_citation_fetch_runs(client, session) -> None:
    session.add(
        CitationFetchRun(
            source="crossref", status="completed", started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc), papers_seen=100, papers_failed=3,
            edges_created=42, edges_already_existed=7,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert len(body["citation_fetch_runs"]) == 1
    run = body["citation_fetch_runs"][0]
    assert run["source"] == "crossref"
    assert run["status"] == "completed"
    assert run["counts"]["papers_seen"] == 100
    assert run["counts"]["papers_failed"] == 3
    assert run["counts"]["edges_created"] == 42
    assert run["counts"]["edges_already_existed"] == 7


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


def test_pipeline_status_reports_papers_by_source(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", source="arxiv")
    _add_paper(session, embedder, "p2", source="arxiv")
    _add_paper(session, embedder, "p3", source="springer")
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["papers_by_source"] == {"arxiv": 2, "springer": 1}


def _add_assessment(session, human_reviewed: bool = False) -> ResearchAssessment:
    research_input = ResearchInput(input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(
        research_input_id=research_input.id, status="completed", human_reviewed=human_reviewed,
    )
    session.add(assessment)
    session.flush()
    return assessment


def test_pipeline_status_reports_assessment_stats(client, session) -> None:
    _add_assessment(session, human_reviewed=False)
    _add_assessment(session, human_reviewed=True)
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    assert body["assessment_stats"] == {"total": 2, "needs_review": 1}


def test_pipeline_status_assessment_stats_collapses_rerun_history(client, session) -> None:
    research_input = ResearchInput(input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    now = datetime.now(timezone.utc)
    session.add(
        ResearchAssessment(
            research_input_id=research_input.id, status="completed", human_reviewed=False,
            created_at=now - timedelta(minutes=5),
        )
    )
    session.add(
        ResearchAssessment(
            research_input_id=research_input.id, status="completed", human_reviewed=True,
            created_at=now,
        )
    )
    session.commit()

    body = client.get("/api/admin/pipeline").json()

    # only the latest (reviewed) counts - the older re-run is superseded
    assert body["assessment_stats"] == {"total": 1, "needs_review": 0}


def test_get_log_returns_the_tail_of_the_pipeline_s_log(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "tail_log", lambda key, lines=200: f"log for {key}, last {lines} lines")

    response = client.get("/api/admin/extraction/log")

    assert response.status_code == 200
    assert response.json() == {"log": "log for extraction, last 200 lines"}


def test_get_log_accepts_a_lines_param(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(routes_module, "tail_log", lambda key, lines=200: calls.append((key, lines)) or "x")

    client.get("/api/admin/embedding/log", params={"lines": 50})

    assert calls == [("embedding", 50)]


def test_get_log_404s_for_an_unknown_pipeline_key(client) -> None:
    response = client.get("/api/admin/bogus_key/log")

    assert response.status_code == 404


def test_stop_marks_the_running_row_stopped(client, session, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    session.add(
        ExtractionRun(extractor_name="hybrid", status="running", started_at=datetime.now(timezone.utc))
    )
    session.commit()
    monkeypatch.setattr(routes_module, "stop", lambda key: True)

    response = client.post("/api/admin/extraction/stop")

    assert response.status_code == 200
    assert response.json() == {"stopped": True, "pipeline": "extraction"}
    run = session.execute(select(ExtractionRun)).scalar_one()
    assert run.status == "stopped"
    assert run.finished_at is not None
    assert run.error_summary == "Stopped by operator"


def test_stop_only_touches_the_matching_ingestion_source(client, session, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    session.add(IngestionRun(source="arxiv", status="running", started_at=datetime.now(timezone.utc)))
    session.add(IngestionRun(source="springer", status="running", started_at=datetime.now(timezone.utc)))
    session.commit()
    monkeypatch.setattr(routes_module, "stop", lambda key: True)

    client.post("/api/admin/ingestion_arxiv/stop")

    runs = {r.source: r.status for r in session.execute(select(IngestionRun)).scalars()}
    assert runs == {"arxiv": "stopped", "springer": "running"}


def test_stop_409s_when_nothing_is_running(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "stop", lambda key: False)

    response = client.post("/api/admin/embedding/stop")

    assert response.status_code == 409


def test_stop_404s_for_an_unknown_pipeline_key(client) -> None:
    response = client.post("/api/admin/bogus_key/stop")

    assert response.status_code == 404


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
        "ingestion_semantic_scholar": False,
        "ingestion_core": False,
        "extraction": False,
        "embedding": False,
        "retrieval_eval": False,
        "extraction_eval": False,
        "citations_fetch": False,
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


def test_trigger_semantic_scholar_ingestion_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post(
        "/api/admin/ingestion/semantic-scholar/run",
        json={"query": '"deep learning"', "max_pages": 4},
    )

    assert calls == [
        (
            "ingestion_semantic_scholar",
            "researchbridge.ingestion.cli_semantic_scholar",
            ["--query", '"deep learning"', "--max-pages", "4"],
        )
    ]


def test_trigger_core_ingestion_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post(
        "/api/admin/ingestion/core/run",
        json={"query": "deep learning", "page_size": 50, "max_pages": 4},
    )

    assert calls == [
        (
            "ingestion_core",
            "researchbridge.ingestion.cli_core",
            ["--query", "deep learning", "--page-size", "50", "--max-pages", "4"],
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


def test_trigger_extraction_with_force_passes_force_flag(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/extraction/run", json={"force": True})

    assert calls == [("extraction", "researchbridge.extraction.cli", ["--force"])]


def test_trigger_embedding_with_force_passes_force_flag(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/embedding/run", json={"limit": 5, "force": True})

    assert calls == [("embedding", "researchbridge.embedding.cli_embed", ["--limit", "5", "--force"])]


def test_trigger_retrieval_eval_calls_trigger_with_no_extra_flags_by_default(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/retrieval-eval/run", json={})

    assert calls == [("retrieval_eval", "researchbridge.retrieval.cli_evaluate", [])]


def test_trigger_retrieval_eval_passes_k_override(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/retrieval-eval/run", json={"k": 5})

    assert calls == [("retrieval_eval", "researchbridge.retrieval.cli_evaluate", ["--k", "5"])]


def test_stop_retrieval_eval_does_not_error_without_a_run_model(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "stop", lambda key: True)

    response = client.post("/api/admin/retrieval_eval/stop")

    assert response.status_code == 200
    assert response.json() == {"stopped": True, "pipeline": "retrieval_eval"}


def test_get_retrieval_eval_returns_unavailable_when_no_results_file(client, monkeypatch, tmp_path) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "RETRIEVAL_EVAL_RESULTS_PATH", tmp_path / "nonexistent.json")

    body = client.get("/api/admin/retrieval-eval").json()

    assert body == {"available": False, "generated_at": None, "k": None, "query_sets": None}


def test_get_retrieval_eval_returns_persisted_results_when_file_exists(client, monkeypatch, tmp_path) -> None:
    import json

    import researchbridge.api.admin_routes as routes_module

    results_path = tmp_path / "retrieval_eval_results.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-28T00:00:00+00:00",
                "k": 10,
                "query_sets": {
                    "self": {
                        "queries": 5,
                        "skipped": 0,
                        "results": [
                            {"method": "tfidf", "precision": 1.0, "recall": 1.0, "ndcg": 1.0, "mrr": 1.0},
                        ],
                    },
                },
            }
        )
    )
    monkeypatch.setattr(routes_module, "RETRIEVAL_EVAL_RESULTS_PATH", results_path)

    body = client.get("/api/admin/retrieval-eval").json()

    assert body["available"] is True
    assert body["k"] == 10
    assert body["query_sets"]["self"]["queries"] == 5
    assert body["query_sets"]["self"]["results"][0]["method"] == "tfidf"


def test_trigger_extraction_eval_calls_trigger_with_no_extra_flags_by_default(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/extraction-eval/run", json={})

    assert calls == [("extraction_eval", "researchbridge.extraction.cli_evaluate", [])]


def test_trigger_extraction_eval_passes_threshold_and_extractor_overrides(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/extraction-eval/run", json={"threshold": 0.6, "extractor": "hybrid"})

    assert calls == [
        (
            "extraction_eval",
            "researchbridge.extraction.cli_evaluate",
            ["--threshold", "0.6", "--extractor", "hybrid"],
        )
    ]


def test_stop_extraction_eval_does_not_error_without_a_run_model(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "stop", lambda key: True)

    response = client.post("/api/admin/extraction_eval/stop")

    assert response.status_code == 200
    assert response.json() == {"stopped": True, "pipeline": "extraction_eval"}


def test_get_extraction_eval_returns_unavailable_when_no_results_file(client, monkeypatch, tmp_path) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(routes_module, "EXTRACTION_EVAL_RESULTS_PATH", tmp_path / "nonexistent.json")

    body = client.get("/api/admin/extraction-eval").json()

    assert body == {"available": False, "generated_at": None, "threshold": None, "paper_count": None, "extractors": None}


def test_get_extraction_eval_returns_persisted_results_when_file_exists(client, monkeypatch, tmp_path) -> None:
    import json

    import researchbridge.api.admin_routes as routes_module

    results_path = tmp_path / "extraction_eval_results.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-28T00:00:00+00:00",
                "threshold": 0.5,
                "paper_count": 10,
                "extractors": {
                    "hybrid": {"problem": {"precision": 0.8, "recall": 0.7, "f1": 0.75}},
                },
            }
        )
    )
    monkeypatch.setattr(routes_module, "EXTRACTION_EVAL_RESULTS_PATH", results_path)

    body = client.get("/api/admin/extraction-eval").json()

    assert body["available"] is True
    assert body["threshold"] == 0.5
    assert body["paper_count"] == 10
    assert body["extractors"]["hybrid"]["problem"]["f1"] == 0.75


def test_trigger_citations_fetch_defaults_to_semantic_scholar(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/citations-fetch/run", json={})

    assert calls == [
        ("citations_fetch", "researchbridge.citations.cli_fetch", ["--all", "--save", "--source", "semantic_scholar"])
    ]


def test_trigger_citations_fetch_passes_crossref_source(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/citations-fetch/run", json={"source": "crossref"})

    assert calls == [
        ("citations_fetch", "researchbridge.citations.cli_fetch", ["--all", "--save", "--source", "crossref"])
    ]


def test_trigger_citations_fetch_passes_force_flag(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/citations-fetch/run", json={"force": True})

    assert calls == [
        (
            "citations_fetch",
            "researchbridge.citations.cli_fetch",
            ["--all", "--save", "--source", "semantic_scholar", "--force"],
        )
    ]


def test_stop_citations_fetch_marks_the_running_row_stopped(client, session, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    session.add(CitationFetchRun(source="crossref", status="running"))
    session.commit()
    monkeypatch.setattr(routes_module, "stop", lambda key: True)

    response = client.post("/api/admin/citations_fetch/stop")

    assert response.status_code == 200
    assert response.json() == {"stopped": True, "pipeline": "citations_fetch"}
    run = session.execute(select(CitationFetchRun)).scalar_one()
    assert run.status == "stopped"
    assert run.finished_at is not None


def test_get_citations_fetch_returns_unavailable_when_no_summary_files(client, monkeypatch, tmp_path) -> None:
    import researchbridge.api.admin_routes as routes_module

    monkeypatch.setattr(
        routes_module,
        "CITATIONS_FETCH_SUMMARY_PATHS",
        {"semantic_scholar": tmp_path / "s2.json", "crossref": tmp_path / "crossref.json"},
    )

    body = client.get("/api/admin/citations-fetch").json()

    unavailable = {
        "available": False,
        "generated_at": None,
        "papers_seen": None,
        "papers_failed": None,
        "edges_created": None,
        "edges_already_existed": None,
    }
    assert body == {"semantic_scholar": unavailable, "crossref": unavailable}


def test_get_citations_fetch_returns_persisted_summary_per_source(client, monkeypatch, tmp_path) -> None:
    import json

    import researchbridge.api.admin_routes as routes_module

    s2_path = tmp_path / "s2.json"
    s2_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-28T00:00:00+00:00",
                "papers_seen": 100,
                "papers_failed": 3,
                "edges_created": 42,
                "edges_already_existed": 7,
            }
        )
    )
    crossref_path = tmp_path / "crossref.json"  # left absent - crossref never run yet
    monkeypatch.setattr(
        routes_module, "CITATIONS_FETCH_SUMMARY_PATHS", {"semantic_scholar": s2_path, "crossref": crossref_path}
    )

    body = client.get("/api/admin/citations-fetch").json()

    assert body["semantic_scholar"]["available"] is True
    assert body["semantic_scholar"]["papers_seen"] == 100
    assert body["semantic_scholar"]["edges_created"] == 42
    assert body["crossref"]["available"] is False


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
        "ingestion_semantic_scholar": False,
        "ingestion_core": False,
        "extraction": True,
        "embedding": False,
        "retrieval_eval": False,
        "extraction_eval": False,
        "citations_fetch": False,
    }


def test_notifications_includes_completed_extraction_run(client, session) -> None:
    session.add(
        ExtractionRun(
            extractor_name="hybrid", status="completed", started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc), papers_processed=10, claims_created=8, candidates_rejected=2,
        )
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert len(body) == 1
    assert body[0]["type"] == "extraction_completed"
    assert body[0]["severity"] == "info"
    assert "8 claims created" in body[0]["message"]
    assert "(forced)" not in body[0]["message"]


def test_notifications_marks_forced_runs(client, session) -> None:
    session.add(
        EmbeddingRun(
            model_name="fake-embedder-v1", status="completed", started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc), papers_processed=5, papers_skipped=0, force=True,
        )
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert "(forced)" in body[0]["message"]


def test_notifications_includes_failed_ingestion_run_as_error_severity(client, session) -> None:
    session.add(
        IngestionRun(
            source="springer", status="failed", started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc), records_fetched=0, records_inserted=0,
            records_duplicate=0, records_failed=0, error_summary="rate limited",
        )
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert body[0]["type"] == "ingestion_failed"
    assert body[0]["severity"] == "error"
    assert "rate limited" in body[0]["message"]


def test_notifications_excludes_still_running_runs(client, session) -> None:
    session.add(
        ExtractionRun(extractor_name="hybrid", status="running", started_at=datetime.now(timezone.utc))
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert body == []


def test_notifications_includes_needs_review_aggregate(client, session) -> None:
    _add_assessment(session, human_reviewed=False)
    _add_assessment(session, human_reviewed=False)
    session.commit()

    body = client.get("/api/admin/notifications").json()

    review_notifications = [n for n in body if n["type"] == "needs_review"]
    assert len(review_notifications) == 1
    assert review_notifications[0]["id"] == "needs_review:2"
    assert "2 assessments need human review" in review_notifications[0]["message"]


def test_notifications_omits_needs_review_when_zero(client, session) -> None:
    _add_assessment(session, human_reviewed=True)
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert not any(n["type"] == "needs_review" for n in body)


def test_notifications_includes_gaps_pending_aggregate(client, session) -> None:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id="p1", title="p1", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    session.add(
        CandidateGap(
            seed_paper_id=paper.id, observation="a pattern", contributing_paper_count=3,
            similarity_threshold=0.8, detection_method="embedding_cosine", status="pending",
        )
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    gap_notifications = [n for n in body if n["type"] == "gaps_pending"]
    assert len(gap_notifications) == 1
    assert gap_notifications[0]["id"] == "gaps_pending:1"


def test_notifications_orders_most_recent_first(client, session) -> None:
    now = datetime.now(timezone.utc)
    session.add(
        ExtractionRun(
            extractor_name="hybrid", status="completed", started_at=now - timedelta(hours=1),
            finished_at=now - timedelta(hours=1), papers_processed=1, claims_created=1, candidates_rejected=0,
        )
    )
    session.add(
        EmbeddingRun(
            model_name="fake-embedder-v1", status="completed", started_at=now,
            finished_at=now, papers_processed=1, papers_skipped=0,
        )
    )
    session.commit()

    body = client.get("/api/admin/notifications").json()

    assert [n["type"] for n in body] == ["embedding_completed", "extraction_completed"]
