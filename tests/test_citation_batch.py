from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.citations.batch import BatchSummary, run_all
from researchbridge.citations.fetch import RawCitationsPayload
from researchbridge.db.models import CitationFetchRun, Paper, PaperCitation


@pytest.fixture()
def session(session_factory):
    from sqlalchemy import text

    s = session_factory()
    # citation_fetch_runs isn't in conftest's TRUNCATE list
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    yield s
    s.execute(text("TRUNCATE TABLE citation_fetch_runs"))
    s.commit()
    s.close()


@dataclass
class FakeFetcher:
    """Returns a fixed payload per source_id, recording every call made."""

    payloads: dict[str, RawCitationsPayload] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def fetch_raw(self, source_id: str) -> RawCitationsPayload:
        self.calls.append(source_id)
        return self.payloads.get(source_id, RawCitationsPayload())


def _add_s2_paper(session, source_id: str) -> Paper:
    paper = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id=source_id, title=f"Paper {source_id}")
    session.add(paper)
    session.commit()
    return paper


def test_run_all_fetches_and_saves_edges_for_every_semantic_scholar_paper(session) -> None:
    a = _add_s2_paper(session, "a")
    b = _add_s2_paper(session, "b")
    fetcher = FakeFetcher(payloads={"a": RawCitationsPayload(cited_source_ids=["b"])})

    summary = run_all(session, fetcher, save=True)

    assert sorted(fetcher.calls) == ["a", "b"]
    assert summary == BatchSummary(papers_seen=2, papers_failed=0, edges_created=1, edges_already_existed=0)
    assert session.query(PaperCitation).count() == 1


def test_run_all_dry_run_does_not_persist(session) -> None:
    _add_s2_paper(session, "a")
    _add_s2_paper(session, "b")
    fetcher = FakeFetcher(payloads={"a": RawCitationsPayload(cited_source_ids=["b"])})

    summary = run_all(session, fetcher, save=False)

    assert summary.edges_created == 1  # what WOULD be created
    assert session.query(PaperCitation).count() == 0  # nothing actually persisted


def test_run_all_updates_the_passed_run_row_incrementally(session, monkeypatch) -> None:
    """The CitationFetchRun row must reflect progress as papers are
    processed, not only once the whole batch finishes - a long-running
    fetch would otherwise show all-zeros the entire time it's running
    (a real bug: the admin UI's live counts never moved)."""
    a = _add_s2_paper(session, "a")
    b = _add_s2_paper(session, "b")
    fetcher = FakeFetcher(payloads={"a": RawCitationsPayload(cited_source_ids=["b"])})
    run = CitationFetchRun(source="semantic_scholar", status="running")
    session.add(run)
    session.commit()

    commits = []
    real_commit = session.commit
    monkeypatch.setattr(session, "commit", lambda: (commits.append(1), real_commit())[-1])

    summary = run_all(session, fetcher, save=True, run=run)

    # at least one commit per paper processed, not just one at the very end
    assert len(commits) >= 2
    assert run.papers_seen == summary.papers_seen == 2
    assert run.edges_created == summary.edges_created == 1
    assert run.edges_already_existed == summary.edges_already_existed == 0
    assert run.papers_failed == summary.papers_failed == 0


def test_run_all_updates_the_passed_run_row_on_a_failing_paper(session) -> None:
    a = _add_s2_paper(session, "a")

    class RaisingFetcher:
        def fetch_raw(self, source_id: str) -> RawCitationsPayload:
            raise RuntimeError("boom")

    run = CitationFetchRun(source="semantic_scholar", status="running")
    session.add(run)
    session.commit()

    run_all(session, RaisingFetcher(), save=True, run=run)

    session.refresh(run)
    assert run.papers_seen == 1
    assert run.papers_failed == 1


def test_run_all_ignores_non_semantic_scholar_papers(session) -> None:
    session.add(Paper(id=uuid.uuid4(), source="arxiv", source_id="x", title="Not S2"))
    session.commit()
    fetcher = FakeFetcher()

    summary = run_all(session, fetcher, save=True)

    assert fetcher.calls == []
    assert summary.papers_seen == 0


def test_run_all_skips_papers_with_existing_outgoing_edges_unless_forced(session) -> None:
    a = _add_s2_paper(session, "a")
    b = _add_s2_paper(session, "b")
    session.add(PaperCitation(citing_paper_id=a.id, cited_paper_id=b.id, source="semantic_scholar", confidence="high"))
    session.commit()
    fetcher = FakeFetcher()

    summary = run_all(session, fetcher, save=True)
    assert sorted(fetcher.calls) == ["b"]  # a already has an outgoing edge, skipped

    fetcher2 = FakeFetcher()
    run_all(session, fetcher2, save=True, force=True)
    assert sorted(fetcher2.calls) == ["a", "b"]  # force reprocesses everything


def test_run_all_continues_past_a_failing_paper(session) -> None:
    a = _add_s2_paper(session, "a")
    _add_s2_paper(session, "b")

    class RaisingFetcher:
        def fetch_raw(self, source_id: str) -> RawCitationsPayload:
            if source_id == a.source_id:
                raise RuntimeError("boom")
            return RawCitationsPayload()

    summary = run_all(session, RaisingFetcher(), save=True)

    assert summary.papers_seen == 2
    assert summary.papers_failed == 1


def _add_doi_paper(session, doi: str, source: str = "springer") -> Paper:
    paper = Paper(id=uuid.uuid4(), source=source, source_id=doi, doi=doi, title=f"Paper {doi}")
    session.add(paper)
    session.commit()
    return paper


def test_run_all_for_crossref_targets_every_doi_bearing_paper_regardless_of_source(session) -> None:
    a = _add_doi_paper(session, "10.1/a", source="arxiv")
    b = _add_doi_paper(session, "10.1/b", source="semantic_scholar")
    session.add(Paper(id=uuid.uuid4(), source="core", source_id="c", title="No DOI"))  # doi=None, skipped
    session.commit()
    fetcher = FakeFetcher(payloads={"10.1/a": RawCitationsPayload(cited_source_ids=["10.1/b"])})

    summary = run_all(session, fetcher, source="crossref", save=True)

    assert sorted(fetcher.calls) == ["10.1/a", "10.1/b"]
    assert summary.papers_seen == 2
    assert session.query(PaperCitation).filter_by(source="crossref").count() == 1


def test_run_all_idempotency_is_scoped_per_source(session) -> None:
    """A paper with an existing semantic_scholar edge must still be
    processed by a crossref --all run, and vice versa - each source tracks
    its own coverage independently."""
    a = _add_doi_paper(session, "10.1/a", source="arxiv")
    b = _add_doi_paper(session, "10.1/b", source="arxiv")
    session.add(PaperCitation(citing_paper_id=a.id, cited_paper_id=b.id, source="semantic_scholar", confidence="high"))
    session.commit()
    fetcher = FakeFetcher()

    summary = run_all(session, fetcher, source="crossref", save=True)

    assert sorted(fetcher.calls) == ["10.1/a", "10.1/b"]  # the semantic_scholar edge doesn't block crossref
    assert summary.papers_seen == 2
