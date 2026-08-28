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
from researchbridge.db.models import Paper, PaperCitation

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    papers_seen: int = 0
    papers_failed: int = 0
    edges_created: int = 0
    edges_already_existed: int = 0


def run_all(session: Session, fetcher, source: str = "semantic_scholar", force: bool = False, save: bool = False) -> BatchSummary:
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
            continue

        attempted = len(resolution.edges)
        if save:
            created = persist_citation_edges(session, resolution, source=source)
            session.commit()
        else:
            created = attempted  # dry run: report what WOULD be created

        summary.edges_created += created
        summary.edges_already_existed += attempted - created

    return summary


def _select_target_papers(session: Session, source: str, force: bool) -> list[Paper]:
    if source == "crossref":
        query = select(Paper).where(Paper.doi.is_not(None))
    else:
        query = select(Paper).where(Paper.source == "semantic_scholar")
    if not force:
        already_done = exists().where(PaperCitation.citing_paper_id == Paper.id, PaperCitation.source == source)
        query = query.where(~already_done)
    return list(session.execute(query).scalars())
