"""Pipeline monitoring (ingestion/extraction/embedding run visibility).

IngestionRun/ExtractionRun/EmbeddingRun have existed since the earliest
migrations but were never exposed anywhere outside direct SQL - this is
read-only, run-level summary visibility only (no per-record error drill-
down, per the "lightweight, not a dashboard" scope decision). Detection/
extraction/embedding themselves stay CLI-triggered, exactly like
gaps_routes.py's detection step - this router only reads what already ran.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.pipeline_triggers import PipelineAlreadyRunning, is_running, tail_log, trigger
from researchbridge.api.schemas import (
    ArxivIngestionTrigger,
    AssessmentStats,
    EmbeddingTrigger,
    ExtractionTrigger,
    Notification,
    PaperExclude,
    PaperSummary,
    PipelineRunOut,
    PipelineStatus,
    PipelineTriggerOut,
    SemanticScholarIngestionTrigger,
    SpringerIngestionTrigger,
)
from researchbridge.api.serializers import to_summary
from researchbridge.db.models import (
    CandidateGap,
    Embedding,
    EmbeddingRun,
    ExtractedClaim,
    ExtractionRun,
    IngestionRun,
    Paper,
    ResearchAssessment,
)

router = APIRouter(prefix="/api/admin")

RECENT_RUNS_LIMIT = 10
PIPELINE_KEYS = ("ingestion_arxiv", "ingestion_springer", "ingestion_semantic_scholar", "extraction", "embedding")

NOTIFICATION_RUNS_PER_TYPE = 15
NOTIFICATION_LIMIT = 30


@router.get("/pipeline", response_model=PipelineStatus)
def pipeline_status(session: Session = Depends(get_session)) -> PipelineStatus:
    total_papers = session.execute(select(func.count(Paper.id))).scalar_one()
    papers_with_claims = session.execute(
        select(func.count(func.distinct(ExtractedClaim.paper_id)))
    ).scalar_one()
    papers_with_embeddings = session.execute(
        select(func.count(func.distinct(Embedding.paper_id)))
    ).scalar_one()
    papers_by_source = dict(
        session.execute(select(Paper.source, func.count(Paper.id)).group_by(Paper.source)).all()
    )

    return PipelineStatus(
        total_papers=total_papers,
        papers_with_claims=papers_with_claims,
        papers_with_embeddings=papers_with_embeddings,
        papers_by_source=papers_by_source,
        assessment_stats=_assessment_stats(session),
        ingestion_runs=[
            _to_run(run, ("records_fetched", "records_inserted", "records_duplicate", "records_failed"))
            for run in _recent(session, IngestionRun)
        ],
        extraction_runs=[
            _to_run(run, ("papers_processed", "claims_created", "candidates_rejected"))
            for run in _recent(session, ExtractionRun)
        ],
        embedding_runs=[
            _to_run(run, ("papers_processed", "papers_skipped")) for run in _recent(session, EmbeddingRun)
        ],
        running={key: is_running(key) for key in PIPELINE_KEYS},
    )


@router.get("/notifications", response_model=list[Notification])
def notifications(session: Session = Depends(get_session)) -> list[Notification]:
    """Everything the admin page's bell icon shows: recently finished runs
    (completed or failed - a still-"running" row isn't a notification yet,
    it's covered by pipeline_status().running), plus the two review queues
    (assessments, candidate gaps) collapsed to one aggregate entry each
    rather than one per pending item.

    No read/unread state lives server-side - id is stable per event so the
    client can track what it has already shown itself (see Notification's
    docstring for how the aggregate ids handle a changing count).
    """
    items: list[Notification] = []

    for run in _recent_finished(session, IngestionRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type="ingestion_failed" if run.status == "failed" else "ingestion_completed",
                severity="error" if run.status == "failed" else "info",
                message=_ingestion_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    for run in _recent_finished(session, ExtractionRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type="extraction_failed" if run.status == "failed" else "extraction_completed",
                severity="error" if run.status == "failed" else "info",
                message=_extraction_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    for run in _recent_finished(session, EmbeddingRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type="embedding_failed" if run.status == "failed" else "embedding_completed",
                severity="error" if run.status == "failed" else "info",
                message=_embedding_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    now = datetime.now(timezone.utc)

    needs_review = _assessment_stats(session).needs_review
    if needs_review > 0:
        items.append(
            Notification(
                id=f"needs_review:{needs_review}",
                type="needs_review",
                severity="info",
                message=f"{needs_review} assessment{'s' if needs_review != 1 else ''} need human review",
                created_at=now,
            )
        )

    gaps_pending = session.execute(
        select(func.count()).select_from(CandidateGap).where(CandidateGap.status == "pending")
    ).scalar_one()
    if gaps_pending > 0:
        items.append(
            Notification(
                id=f"gaps_pending:{gaps_pending}",
                type="gaps_pending",
                severity="info",
                message=f"{gaps_pending} candidate gap{'s' if gaps_pending != 1 else ''} pending review",
                created_at=now,
            )
        )

    items.sort(key=lambda n: n.created_at, reverse=True)
    return items[:NOTIFICATION_LIMIT]


def _recent_finished(session: Session, model: type) -> list:
    return list(
        session.execute(
            select(model)
            .where(model.status.in_(("completed", "failed")))
            .order_by(model.started_at.desc())
            .limit(NOTIFICATION_RUNS_PER_TYPE)
        ).scalars()
    )


def _ingestion_message(run: IngestionRun) -> str:
    label = (run.source or "ingestion").replace("_", " ")
    if run.status == "failed":
        return f"{label} ingestion failed: {run.error_summary or 'unknown error'}"
    return (
        f"{label} ingestion completed: {run.records_inserted} inserted, "
        f"{run.records_duplicate} duplicate, {run.records_failed} failed"
    )


def _extraction_message(run: ExtractionRun) -> str:
    forced = " (forced)" if run.force else ""
    if run.status == "failed":
        return f"Extraction run failed{forced}: {run.error_summary or 'unknown error'}"
    return f"Extraction run completed{forced}: {run.claims_created} claims created, {run.candidates_rejected} rejected"


def _embedding_message(run: EmbeddingRun) -> str:
    forced = " (forced)" if run.force else ""
    if run.status == "failed":
        return f"Embedding run failed{forced}: {run.error_summary or 'unknown error'}"
    return f"Embedding run completed{forced}: {run.papers_processed} processed, {run.papers_skipped} skipped"


def _assessment_stats(session: Session) -> AssessmentStats:
    """The latest assessment per research_input - mirrors GET /api/assessments'
    collapse-to-latest query (assessment_routes.py), just counts instead of a
    full listing, so re-run history doesn't inflate these numbers."""
    latest = (
        select(
            ResearchAssessment.research_input_id,
            func.max(ResearchAssessment.created_at).label("max_created_at"),
        )
        .group_by(ResearchAssessment.research_input_id)
        .subquery()
    )
    latest_ids = select(ResearchAssessment.id).join(
        latest,
        and_(
            ResearchAssessment.research_input_id == latest.c.research_input_id,
            ResearchAssessment.created_at == latest.c.max_created_at,
        ),
    )

    total = session.execute(select(func.count()).select_from(latest_ids.subquery())).scalar_one()
    needs_review = session.execute(
        select(func.count())
        .select_from(ResearchAssessment)
        .where(ResearchAssessment.id.in_(latest_ids), ResearchAssessment.human_reviewed.is_(False))
    ).scalar_one()

    return AssessmentStats(total=total, needs_review=needs_review)


def _recent(session: Session, model: type) -> list:
    return list(
        session.execute(select(model).order_by(model.started_at.desc()).limit(RECENT_RUNS_LIMIT)).scalars()
    )


def _to_run(run, count_fields: tuple[str, ...]) -> PipelineRunOut:
    return PipelineRunOut(
        id=run.id,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_summary=run.error_summary,
        source=getattr(run, "source", None),
        counts={field: getattr(run, field) for field in count_fields},
    )


@router.get("/{key}/log")
def get_pipeline_log(key: str, lines: int = Query(200, ge=1, le=2000)) -> dict:
    """The tail of the given pipeline's most recent log file - polled by the
    admin page while that pipeline is running (see pipeline_triggers.tail_log)."""
    if key not in PIPELINE_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline key {key!r}")
    return {"log": tail_log(key, lines=lines)}


@router.put("/papers/{paper_id}/exclude", response_model=PaperSummary)
def exclude_paper(
    paper_id: uuid.UUID, payload: PaperExclude, session: Session = Depends(get_session)
) -> PaperSummary:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")

    paper.excluded_at = datetime.now(timezone.utc) if payload.excluded else None
    session.commit()

    return to_summary(session, paper)


def _trigger_or_409(key: str, module: str, args: list[str]) -> PipelineTriggerOut:
    try:
        log_path = trigger(key, module, args)
    except PipelineAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PipelineTriggerOut(started=True, pipeline=key, log_file=str(log_path))


@router.post("/ingestion/arxiv/run", response_model=PipelineTriggerOut)
def trigger_arxiv_ingestion(payload: ArxivIngestionTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.search_query is not None:
        args += ["--search-query", payload.search_query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409("ingestion_arxiv", "researchbridge.ingestion.cli", args)


@router.post("/ingestion/springer/run", response_model=PipelineTriggerOut)
def trigger_springer_ingestion(payload: SpringerIngestionTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409("ingestion_springer", "researchbridge.ingestion.cli_springer", args)


@router.post("/ingestion/semantic-scholar/run", response_model=PipelineTriggerOut)
def trigger_semantic_scholar_ingestion(payload: SemanticScholarIngestionTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409(
        "ingestion_semantic_scholar", "researchbridge.ingestion.cli_semantic_scholar", args
    )


@router.post("/extraction/run", response_model=PipelineTriggerOut)
def trigger_extraction(payload: ExtractionTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.extractor is not None:
        args += ["--extractor", payload.extractor]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409("extraction", "researchbridge.extraction.cli", args)


@router.post("/embedding/run", response_model=PipelineTriggerOut)
def trigger_embedding(payload: EmbeddingTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409("embedding", "researchbridge.embedding.cli_embed", args)
