from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pytest
from sqlalchemy import text

from researchbridge.citations.cli_fetch import _run_batch
from researchbridge.citations.fetch import RawCitationsPayload
from researchbridge.db.models import CitationFetchRun


@pytest.fixture()
def session(session_factory):
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
