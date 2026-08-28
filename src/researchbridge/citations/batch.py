"""Batch citation fetching over every Semantic Scholar paper in the corpus.

Idempotent by default: a paper with at least one outgoing PaperCitation edge
is skipped on the next run (--force reprocesses it) - same
"no new tracking table, reuse what already answers the question" principle
gaps/batch.py already applies to candidate_gaps.seed_paper_id.
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


def run_all(session: Session, fetcher, force: bool = False, save: bool = False) -> BatchSummary:
    summary = BatchSummary()

    for paper in _select_target_papers(session, force):
        summary.papers_seen += 1
        try:
            payload = fetcher.fetch_raw(paper.source_id)
            resolution = resolve_citations(session, paper, payload)
        except Exception:
            logger.exception("Citation fetch failed for paper %s", paper.id)
            summary.papers_failed += 1
            continue

        attempted = len(resolution.edges)
        if save:
            created = persist_citation_edges(session, resolution)
            session.commit()
        else:
            created = attempted  # dry run: report what WOULD be created

        summary.edges_created += created
        summary.edges_already_existed += attempted - created

    return summary


def _select_target_papers(session: Session, force: bool) -> list[Paper]:
    query = select(Paper).where(Paper.source == "semantic_scholar")
    if not force:
        already_done = exists().where(
            PaperCitation.citing_paper_id == Paper.id, PaperCitation.source == "semantic_scholar"
        )
        query = query.where(~already_done)
    return list(session.execute(query).scalars())
