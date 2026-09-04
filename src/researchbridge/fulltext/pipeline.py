"""Fetch + parse + persist full text for open-access papers (Sec 46).

Structurally mirrors extraction/pipeline.py: idempotent paper selection,
per-paper try/except that logs to fulltext_fetch_errors and continues
rather than crashing the run, incremental run-history updates.

No consumer yet - extraction/pipeline.py and every Extractor implementation
stay abstract-only. Wiring full text into extraction is a separate, future
slice (see the design spec's "Two-slice split").
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText
from researchbridge.fulltext.parse import PdfParseError, parse_pdf
from researchbridge.fulltext.pdf_url import resolve_pdf_url

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


class FullTextFetchPipeline:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def run(self, limit: int | None = None, force: bool = False) -> str:
        session = self.session_factory()
        run = FullTextFetchRun(status="running", force=force)
        session.add(run)
        session.commit()
        run_id = str(run.id)

        try:
            papers = self._select_papers(session, limit, force)
            total = len(papers)
            logger.info("Full-text fetch run %s starting: %d paper(s) to process", run_id, total)

            for i, paper in enumerate(papers, start=1):
                self._process_paper(session, run, paper, force)
                run.papers_seen += 1
                session.commit()
                if i % 10 == 0 or i == total:
                    logger.info(
                        "Full-text fetch run %s: %d/%d papers seen (fetched=%d, failed=%d)",
                        run_id, i, total, run.papers_fetched, run.papers_failed,
                    )

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            session.commit()
            logger.info(
                "Full-text fetch run %s completed: %d fetched, %d failed", run_id, run.papers_fetched, run.papers_failed
            )

        except Exception as exc:  # noqa: BLE001 - run must record failure, not crash silently
            logger.exception("Full-text fetch run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id

    def _process_paper(self, session: Session, run: FullTextFetchRun, paper: Paper, force: bool) -> None:
        url = resolve_pdf_url(paper)
        if url is None:
            run.papers_skipped_no_url += 1
            return

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Full-text fetch failed for paper %s: %s", paper.id, exc)
            self._record_error(session, run.id, paper.id, "fetch_error", str(exc)[:2000])
            run.papers_failed += 1
            return

        try:
            sections = parse_pdf(response.content)
        except PdfParseError as exc:
            logger.warning("Full-text parse failed for paper %s: %s", paper.id, exc)
            self._record_error(session, run.id, paper.id, "parse_error", str(exc)[:2000])
            run.papers_failed += 1
            return

        existing = session.execute(select(PaperFullText).where(PaperFullText.paper_id == paper.id)).scalar_one_or_none()
        if existing is not None:
            existing.sections = sections
            existing.source_url = url
        else:
            session.add(PaperFullText(paper_id=paper.id, sections=sections, source_url=url))
        run.papers_fetched += 1

    def _select_papers(self, session: Session, limit: int | None, force: bool) -> list[Paper]:
        query = select(Paper).where(Paper.open_access.is_(True))
        if not force:
            already_fetched = select(PaperFullText.paper_id)
            query = query.where(Paper.id.notin_(already_fetched))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())

    def _record_error(self, session: Session, run_id: Any, paper_id: Any, error_type: str, detail: str) -> None:
        session.add(
            FullTextFetchError(
                fulltext_fetch_run_id=run_id, paper_id=paper_id, error_type=error_type, error_detail=detail,
            )
        )
