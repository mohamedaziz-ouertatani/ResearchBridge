from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.benchmark_routes import get_benchmark_dir
from researchbridge.benchmark.annotation_template import render_annotation_template
from researchbridge.benchmark.store import load
from researchbridge.db.models import Paper


def _template(source_id: str, domain: str = "NLP") -> str:
    paper = Paper(
        id=uuid.uuid4(),
        source="arxiv",
        source_id=source_id,
        title=f"Paper {source_id}",
        publication_date=date(2024, 3, 1),
        url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={},
        ingestion_metadata={},
    )
    return render_annotation_template(paper, domain=domain)


@pytest.fixture()
def benchmark_dir(tmp_path) -> Path:
    annotations = tmp_path / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "arxiv_1111.11111.yaml").write_text(_template("1111.11111"), encoding="utf-8")
    (annotations / "arxiv_2222.22222.yaml").write_text(
        _template("2222.22222", domain="Systems"), encoding="utf-8"
    )

    fulltext = tmp_path / "fulltext"
    fulltext.mkdir()
    (fulltext / "1111.11111.txt").write_text("Full text of the first paper.", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(benchmark_dir):
    app = create_app()
    app.dependency_overrides[get_benchmark_dir] = lambda: benchmark_dir
    with TestClient(app) as c:
        yield c


def test_lists_every_annotation_with_progress(client) -> None:
    body = client.get("/api/benchmark/papers").json()

    assert len(body) == 2
    by_id = {row["source_id"]: row for row in body}
    assert by_id["1111.11111"]["filled"] == 0
    assert by_id["1111.11111"]["total"] == 10
    assert by_id["1111.11111"]["is_complete"] is False
    assert by_id["2222.22222"]["domain"] == "Systems"


def test_list_reports_which_papers_have_fulltext(client) -> None:
    by_id = {row["source_id"]: row for row in client.get("/api/benchmark/papers").json()}

    assert by_id["1111.11111"]["has_fulltext"] is True
    assert by_id["2222.22222"]["has_fulltext"] is False  # no cached text for this one


def test_detail_includes_fulltext_and_empty_fields(client) -> None:
    body = client.get("/api/benchmark/papers/1111.11111").json()

    assert body["fulltext"] == "Full text of the first paper."
    assert body["fields"]["problem"] == ""
    assert body["research_gap"] == {"addressed": "", "remaining": ""}
    assert body["key_evidence"] == []


def test_detail_without_fulltext_returns_null(client) -> None:
    assert client.get("/api/benchmark/papers/2222.22222").json()["fulltext"] is None


def test_detail_404s_for_unknown_paper(client) -> None:
    assert client.get("/api/benchmark/papers/9999.99999").status_code == 404


def test_update_persists_to_the_yaml_file(client, benchmark_dir) -> None:
    response = client.put(
        "/api/benchmark/papers/1111.11111",
        json={"problem": "Retrieval degrades on long documents.", "method": "Two-stage reranking."},
    )

    assert response.status_code == 200
    assert response.json()["filled"] == 2

    on_disk = load(benchmark_dir / "annotations" / "arxiv_1111.11111.yaml")
    assert on_disk.fields["problem"] == "Retrieval degrades on long documents."
    assert on_disk.fields["method"] == "Two-stage reranking."


def test_partial_update_leaves_other_fields_untouched(client, benchmark_dir) -> None:
    client.put("/api/benchmark/papers/1111.11111", json={"problem": "first"})
    client.put("/api/benchmark/papers/1111.11111", json={"method": "second"})

    on_disk = load(benchmark_dir / "annotations" / "arxiv_1111.11111.yaml")
    assert on_disk.fields["problem"] == "first"  # not cleared by the second write
    assert on_disk.fields["method"] == "second"


def test_update_saves_research_gap_and_evidence(client, benchmark_dir) -> None:
    client.put(
        "/api/benchmark/papers/1111.11111",
        json={
            "research_gap": {"addressed": "No prior benchmark.", "remaining": "Still untested at scale."},
            "key_evidence": [{"text": "We observe a 12% drop.", "section": "Results"}],
        },
    )

    on_disk = load(benchmark_dir / "annotations" / "arxiv_1111.11111.yaml")
    assert on_disk.research_gap["remaining"] == "Still untested at scale."
    assert on_disk.key_evidence == [{"text": "We observe a 12% drop.", "section": "Results"}]


def test_update_cannot_rewrite_identity(client, benchmark_dir) -> None:
    client.put("/api/benchmark/papers/1111.11111", json={"source_id": "9999.99999", "problem": "kept"})

    on_disk = load(benchmark_dir / "annotations" / "arxiv_1111.11111.yaml")
    assert on_disk.source_id == "1111.11111"
    assert on_disk.fields["problem"] == "kept"


def test_progress_aggregates_across_papers(client) -> None:
    client.put("/api/benchmark/papers/1111.11111", json={"problem": "a", "method": "b"})

    body = client.get("/api/benchmark/progress").json()

    assert body["papers"] == 2
    assert body["complete"] == 0
    assert body["fields_filled"] == 2
    assert body["fields_total"] == 20  # 2 papers x 10 fields
