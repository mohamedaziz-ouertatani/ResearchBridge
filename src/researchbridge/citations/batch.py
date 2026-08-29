"""Batch citation fetching over every paper the given source can cover:
every source="semantic_scholar" paper for Semantic Scholar, every
DOI-bearing paper (any connector) for CrossRef.

Idempotent by default, scoped per source: a paper with at least one
outgoing PaperCitation edge FROM THAT SOURCE is skipped on the next run
of that source (--force reprocesses it) - a paper already covered by
Semantic Scholar still gets its own CrossRef pass, since the two sources
have independent coverage. Same "no new tracking table, reuse what
already answers the question" principle gaps/batch.py already applies to
candidate_gaps.seed_paper_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from researchbridge.citations.fetch import persist_citation_edges, resolve_citations
from researchbridge.db.models import CitationFetchRun, Paper, PaperCitation

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    papers_seen: int = 0
    papers_failed: int = 0
    edges_created: int = 0
    edges_already_existed: int = 0


def run_all(
    session: Session,
    fetcher,
    source: str = "semantic_scholar",
    force: bool = False,
    save: bool = False,
    run: CitationFetchRun | None = None,
) -> BatchSummary:
    """run: an optional in-progress CitationFetchRun row to update after
    every paper, not just once the whole batch finishes - without this, a
    long-running fetch (rate-limited to 1-6s/request, easily tens of
    thousands of papers) would show all-zeros in the admin UI the entire
    time it's running, only jumping to real numbers at the very end."""
    summary = BatchSummary()

    for paper in _select_target_papers(session, source, force):
        summary.papers_seen += 1
        try:
            identifier = paper.source_id if source == "semantic_scholar" else paper.doi
            payload = fetcher.fetch_raw(identifier)
            resolution = resolve_citations(session, paper, payload, source=source)
        except Exception:
            logger.exception("Citation fetch failed for paper %s", paper.id)
            summary.papers_failed += 1
            _sync_run(session, run, summary)
            continue

        attempted = len(resolution.edges)
        if save:
            created = persist_citation_edges(session, resolution, source=source)
        else:
            created = attempted  # dry run: report what WOULD be created

        summary.edges_created += created
        summary.edges_already_existed += attempted - created
        _sync_run(session, run, summary)

    return summary


def _sync_run(session: Session, run: CitationFetchRun | None, summary: BatchSummary) -> None:
    if run is None:
        session.commit()  # still commit any edges persisted this iteration
        return
    run.papers_seen = summary.papers_seen
    run.papers_failed = summary.papers_failed
    run.edges_created = summary.edges_created
    run.edges_already_existed = summary.edges_already_existed
    session.commit()  # updates both the run row and any edges persisted this iteration


def _select_target_papers(session: Session, source: str, force: bool) -> list[Paper]:
    if source == "crossref":
        query = select(Paper).where(Paper.doi.is_not(None))
    else:
        query = select(Paper).where(Paper.source == "semantic_scholar")
    if not force:
        already_done = exists().where(PaperCitation.citing_paper_id == Paper.id, PaperCitation.source == source)
        query = query.where(~already_done)
    return list(session.execute(query).scalars())
