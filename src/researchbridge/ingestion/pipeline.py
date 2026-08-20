"""Ingestion pipeline: fetch -> normalize -> validate -> deduplicate -> persist.

Each stage is a distinct, separately testable step. Progress (including
resume_state and running counters) is written to `ingestion_runs`
incrementally, so a crashed run leaves an accurate, resumable record rather
than silence.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.connectors.base import Connector, NormalizedPaper
from researchbridge.db.models import IngestionError, IngestionRun, Paper

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, connector: Connector, session_factory: sessionmaker[Session]) -> None:
        self.connector = connector
        self.session_factory = session_factory

    def run(
        self,
        resume_state: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> str:
        """Run ingestion to completion (or until max_pages), returning the run id.

        Starting resume_state can be passed explicitly to resume a prior run
        (e.g. read from a previous ingestion_runs row); defaults to starting
        from scratch.
        """
        session = self.session_factory()
        run = IngestionRun(source=self.connector.source_name, status="running", resume_state=resume_state)
        session.add(run)
        session.commit()
        run_id = str(run.id)

        state = resume_state
        pages_fetched = 0

        try:
            while True:
                fetch_result = self.connector.fetch(state)

                normalized = [_normalize(p) for p in fetch_result.papers]
                valid, invalid = _validate(normalized)
                unique, duplicates = _deduplicate(session, valid)
                inserted, persist_failures = _persist(session, unique)

                for paper, detail in invalid:
                    _record_error(session, run.id, self.connector.source_name, paper, "validation_error", detail)
                for paper, detail in persist_failures:
                    _record_error(session, run.id, self.connector.source_name, paper, "db_error", detail)

                run.records_fetched += len(fetch_result.papers)
                run.records_inserted += len(inserted)
                run.records_duplicate += len(duplicates)
                run.records_failed += len(invalid) + len(persist_failures)
                run.resume_state = fetch_result.resume_state
                session.commit()

                state = fetch_result.resume_state
                pages_fetched += 1

                if fetch_result.exhausted:
                    run.status = "completed"
                    run.finished_at = datetime.now(UTC)
                    session.commit()
                    break

                if max_pages is not None and pages_fetched >= max_pages:
                    run.status = "paused"
                    session.commit()
                    break

        except Exception as exc:  # noqa: BLE001 - deliberately broad: run must record failure, not crash silently
            logger.exception("Ingestion run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id


def _normalize(paper: NormalizedPaper) -> NormalizedPaper:
    """Explicit pass-through stage.

    Connectors currently return already-normalized NormalizedPaper objects,
    so there is nothing to transform yet. This stage exists so a future
    connector that yields raw provider dicts still fits the same
    fetch -> normalize -> validate -> deduplicate -> persist shape without
    restructuring the pipeline.
    """
    return paper


def _validate(
    papers: list[NormalizedPaper],
) -> tuple[list[NormalizedPaper], list[tuple[NormalizedPaper, str]]]:
    valid: list[NormalizedPaper] = []
    invalid: list[tuple[NormalizedPaper, str]] = []
    for paper in papers:
        if not paper.source_id or not paper.source_id.strip():
            invalid.append((paper, "missing source_id"))
            continue
        if not paper.title or not paper.title.strip():
            invalid.append((paper, "missing title"))
            continue
        valid.append(paper)
    return valid, invalid


def _deduplicate(
    session: Session, papers: list[NormalizedPaper]
) -> tuple[list[NormalizedPaper], list[NormalizedPaper]]:
    if not papers:
        return [], []

    source = papers[0].source
    candidate_ids = [p.source_id for p in papers]
    existing_rows = session.execute(
        select(Paper.source_id).where(Paper.source == source, Paper.source_id.in_(candidate_ids))
    ).scalars()
    existing_ids = set(existing_rows)

    unique: list[NormalizedPaper] = []
    duplicates: list[NormalizedPaper] = []
    seen_in_batch: set[str] = set()
    for paper in papers:
        if paper.source_id in existing_ids or paper.source_id in seen_in_batch:
            duplicates.append(paper)
            continue
        seen_in_batch.add(paper.source_id)
        unique.append(paper)
    return unique, duplicates


def _persist(
    session: Session, papers: list[NormalizedPaper]
) -> tuple[list[Paper], list[tuple[NormalizedPaper, str]]]:
    inserted: list[Paper] = []
    failures: list[tuple[NormalizedPaper, str]] = []

    for paper in papers:
        try:
            with session.begin_nested():
                row = Paper(
                    source=paper.source,
                    source_id=paper.source_id,
                    doi=paper.doi,
                    title=paper.title,
                    abstract=paper.abstract,
                    publication_date=paper.publication_date,
                    venue=paper.venue,
                    document_type=paper.document_type,
                    language=paper.language,
                    url=paper.url,
                    open_access=paper.open_access,
                    raw_metadata=paper.raw_metadata,
                )
                session.add(row)
            inserted.append(row)
        except IntegrityError as exc:
            failures.append((paper, str(exc)[:2000]))

    return inserted, failures


def _record_error(
    session: Session,
    run_id: Any,
    source: str,
    paper: NormalizedPaper,
    error_type: str,
    detail: str,
) -> None:
    session.add(
        IngestionError(
            ingestion_run_id=run_id,
            source=source,
            raw_payload=_paper_to_json(paper),
            error_type=error_type,
            error_detail=detail,
        )
    )


def _paper_to_json(paper: NormalizedPaper) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, date | datetime):
            return value.isoformat()
        return value

    raw = dataclasses.asdict(paper)
    return {k: convert(v) for k, v in raw.items()}
