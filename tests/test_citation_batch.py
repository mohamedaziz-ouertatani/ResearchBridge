from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.citations.batch import BatchSummary, run_all
from researchbridge.citations.fetch import RawCitationsPayload
from researchbridge.db.models import Paper, PaperCitation


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
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
