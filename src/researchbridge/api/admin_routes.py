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
    return _trigger_or_409("extraction", "researchbridge.extraction.cli", args)


@router.post("/embedding/run", response_model=PipelineTriggerOut)
def trigger_embedding(payload: EmbeddingTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    return _trigger_or_409("embedding", "researchbridge.embedding.cli_embed", args)
