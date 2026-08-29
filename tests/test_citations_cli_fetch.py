from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import pytest
from sqlalchemy import text

from researchbridge.citations.batch import BatchSummary
from researchbridge.citations.cli_fetch import (
    SUMMARY_PATH_BY_SOURCE,
    _build_summary_json,
    _run_batch,
    summary_path_for,
    write_summary_json,
)
from researchbridge.citations.fetch import RawCitationsPayload
from researchbridge.db.models import CitationFetchRun


def test_build_summary_json_shapes_batch_summary() -> None:
    summary = BatchSummary(papers_seen=10, papers_failed=1, edges_created=5, edges_already_existed=2)

    result = _build_summary_json(summary)

    assert "generated_at" in result
    assert result["papers_seen"] == 10
    assert result["papers_failed"] == 1
    assert result["edges_created"] == 5
    assert result["edges_already_existed"] == 2


def test_write_summary_json_creates_parent_dir_and_writes_valid_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "citations_fetch_summary.json"
    summary = BatchSummary(papers_seen=3, papers_failed=0, edges_created=1, edges_already_existed=0)

    write_summary_json(output_path, summary)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["papers_seen"] == 3
    assert written["edges_created"] == 1


@pytest.fixture()
def session(session_factory, tmp_path, monkeypatch):
    import researchbridge.citations.cli_fetch as cli_fetch_module

    monkeypatch.setattr(
        cli_fetch_module,
        "SUMMARY_PATH_BY_SOURCE",
        {"semantic_scholar": tmp_path / "s2.json", "crossref": tmp_path / "crossref.json"},
    )
    s = session_factory()
    # citation_fetch_runs isn't in conftest's TRUNCATE list - clear it before
    # AND after, since test order across files isn't guaranteed.
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    yield s
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    s.close()


@dataclass
class FakeFetcher:
    payloads: dict[str, RawCitationsPayload] = field(default_factory=dict)

    def fetch_raw(self, source_id: str) -> RawCitationsPayload:
        return self.payloads.get(source_id, RawCitationsPayload())


def test_run_batch_creates_a_completed_run_row(session) -> None:
    args = argparse.Namespace(source="crossref", force=False, save=True)

    _run_batch(session, FakeFetcher(), args)

    run = session.query(CitationFetchRun).one()
    assert run.source == "crossref"
    assert run.status == "completed"
    assert run.finished_at is not None
    assert run.papers_seen == 0  # no papers in the corpus for this test


def test_run_batch_marks_the_row_failed_on_exception(session, monkeypatch) -> None:
    import researchbridge.citations.cli_fetch as cli_fetch_module

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_fetch_module, "run_all", _raise)
    args = argparse.Namespace(source="semantic_scholar", force=False, save=True)

    with pytest.raises(RuntimeError, match="boom"):
        _run_batch(session, FakeFetcher(), args)

    run = session.query(CitationFetchRun).one()
    assert run.status == "failed"
    assert run.error_summary == "boom"
    assert run.finished_at is not None


def test_summary_path_for_gives_each_source_its_own_file() -> None:
    """Semantic Scholar and CrossRef must not clobber each other's last-run
    summary - they're independent sources with independent coverage."""
    assert summary_path_for("semantic_scholar") != summary_path_for("crossref")
    assert summary_path_for("semantic_scholar") == SUMMARY_PATH_BY_SOURCE["semantic_scholar"]
    assert summary_path_for("crossref") == SUMMARY_PATH_BY_SOURCE["crossref"]
