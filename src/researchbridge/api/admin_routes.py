"""Pipeline monitoring (ingestion/extraction/embedding run visibility).

IngestionRun/ExtractionRun/EmbeddingRun have existed since the earliest
migrations but were never exposed anywhere outside direct SQL - this is
read-only, run-level summary visibility only (no per-record error drill-
down, per the "lightweight, not a dashboard" scope decision). Detection/
extraction/embedding themselves stay CLI-triggered, exactly like
gaps_routes.py's detection step - this router only reads what already ran.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.pipeline_triggers import PipelineAlreadyRunning, is_running, stop, tail_log, trigger
from researchbridge.api.schemas import (
    ArxivIngestionTrigger,
    AssessmentStats,
    CitationSourceSummary,
    CitationsFetchOut,
    CitationsFetchTrigger,
    CoreIngestionTrigger,
    CorpusHealth,
    EmbeddingTrigger,
    ExtractionEvalOut,
    ExtractionEvalTrigger,
    ExtractionTrigger,
    GapReviewStats,
    Notification,
    PaperExclude,
    PaperSummary,
    PipelineRunOut,
    PipelineStatus,
    PipelineStopOut,
    PipelineTriggerOut,
    RetrievalEvalOut,
    RetrievalEvalTrigger,
    SemanticScholarIngestionTrigger,
    SpringerIngestionTrigger,
)
from researchbridge.api.serializers import to_summary
from researchbridge.citations.cli_fetch import SUMMARY_PATH_BY_SOURCE as CITATIONS_FETCH_SUMMARY_PATHS
from researchbridge.db.models import (
    CandidateGap,
    CitationFetchRun,
    Embedding,
    EmbeddingRun,
    ExtractedClaim,
    ExtractionRun,
    IngestionError,
    IngestionRun,
    Paper,
    PaperCitation,
    ResearchAssessment,
)
from researchbridge.extraction.cli_evaluate import DEFAULT_RESULTS_PATH as EXTRACTION_EVAL_RESULTS_PATH
from researchbridge.retrieval.cli_evaluate import DEFAULT_RESULTS_PATH as RETRIEVAL_EVAL_RESULTS_PATH

router = APIRouter(prefix="/api/admin")

RECENT_RUNS_LIMIT = 10
PIPELINE_KEYS = (
    "ingestion_arxiv",
    "ingestion_springer",
    "ingestion_semantic_scholar",
    "ingestion_core",
    "extraction",
    "embedding",
    "retrieval_eval",
    "extraction_eval",
    "citations_fetch",
)

NOTIFICATION_RUNS_PER_TYPE = 15
NOTIFICATION_LIMIT = 30

# Which *_runs table (and, for ingestion, which source) a pipeline key's
# in-progress row lives in - used only by stop_pipeline() to mark that row
# "stopped" once the subprocess is killed, since the subprocess itself never
# gets a chance to do that for us. "retrieval_eval" has no entry here - it's
# a one-off diagnostic (see RetrievalEvalOut docstring), not a repeating
# pipeline stage, so there's no *_runs row to mark; stop_pipeline() treats a
# missing entry as "nothing to update in the DB" rather than erroring.
RUN_MODEL_BY_KEY: dict[str, tuple[type, str | None]] = {
    "ingestion_arxiv": (IngestionRun, "arxiv"),
    "ingestion_springer": (IngestionRun, "springer"),
    "ingestion_semantic_scholar": (IngestionRun, "semantic_scholar"),
    "ingestion_core": (IngestionRun, "core"),
    "extraction": (ExtractionRun, None),
    "embedding": (EmbeddingRun, None),
    "citations_fetch": (CitationFetchRun, None),
}


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
        corpus_health=_corpus_health(session),
        assessment_stats=_assessment_stats(session),
        gap_stats=_gap_stats(session),
        ingestion_errors_by_type=_ingestion_errors_by_type(session),
        ingestion_runs=[
            _to_run(run, ("records_fetched", "records_inserted", "records_duplicate", "records_failed"))
            for run in _recent(session, IngestionRun)
        ],
        extraction_runs=[
            _to_run(run, ("papers_processed", "claims_created", "candidates_rejected"))
            for run in _recent(session, ExtractionRun)
        ],
        citation_fetch_runs=[
            _to_run(run, ("papers_seen", "papers_failed", "edges_created", "edges_already_existed"))
            for run in _recent(session, CitationFetchRun)
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
                type=f"ingestion_{run.status}",
                severity=_severity(run.status),
                message=_ingestion_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    for run in _recent_finished(session, ExtractionRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type=f"extraction_{run.status}",
                severity=_severity(run.status),
                message=_extraction_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    for run in _recent_finished(session, EmbeddingRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type=f"embedding_{run.status}",
                severity=_severity(run.status),
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
            .where(model.status.in_(("completed", "failed", "stopped")))
            .order_by(model.started_at.desc())
            .limit(NOTIFICATION_RUNS_PER_TYPE)
        ).scalars()
    )


def _severity(status: str) -> str:
    return "error" if status == "failed" else "info"


def _ingestion_message(run: IngestionRun) -> str:
    label = (run.source or "ingestion").replace("_", " ")
    if run.status == "failed":
        return f"{label} ingestion failed: {run.error_summary or 'unknown error'}"
    if run.status == "stopped":
        return f"{label} ingestion stopped: {run.records_inserted} inserted so far"
    return (
        f"{label} ingestion completed: {run.records_inserted} inserted, "
        f"{run.records_duplicate} duplicate, {run.records_failed} failed"
    )


def _extraction_message(run: ExtractionRun) -> str:
    forced = " (forced)" if run.force else ""
    if run.status == "failed":
        return f"Extraction run failed{forced}: {run.error_summary or 'unknown error'}"
    if run.status == "stopped":
        return f"Extraction run stopped{forced}: {run.claims_created} claims created so far"
    return f"Extraction run completed{forced}: {run.claims_created} claims created, {run.candidates_rejected} rejected"


def _embedding_message(run: EmbeddingRun) -> str:
    forced = " (forced)" if run.force else ""
    if run.status == "failed":
        return f"Embedding run failed{forced}: {run.error_summary or 'unknown error'}"
    if run.status == "stopped":
        return f"Embedding run stopped{forced}: {run.papers_processed} processed so far"
    return f"Embedding run completed{forced}: {run.papers_processed} processed, {run.papers_skipped} skipped"


def _corpus_health(session: Session) -> CorpusHealth:
    """Cheap count queries flagging papers stuck mid-pipeline or unreachable
    by a citation source - see CorpusHealth's field docstrings for what each
    one means. No per-paper listing, matching this router's existing
    "lightweight, not a dashboard" scope."""
    missing_doi = session.execute(
        select(func.count()).select_from(Paper).where(Paper.doi.is_(None))
    ).scalar_one()

    excluded = session.execute(
        select(func.count()).select_from(Paper).where(Paper.excluded_at.is_not(None))
    ).scalar_one()

    has_embedding = exists().where(Embedding.paper_id == Paper.id)
    claims_without_embeddings = session.execute(
        select(func.count(func.distinct(ExtractedClaim.paper_id)))
        .select_from(ExtractedClaim)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(~has_embedding)
    ).scalar_one()

    eligible_for_citations = Paper.doi.is_not(None) | (Paper.source == "semantic_scholar")
    has_citation_edge = exists().where(PaperCitation.citing_paper_id == Paper.id)
    no_citation_coverage = session.execute(
        select(func.count()).select_from(Paper).where(eligible_for_citations, ~has_citation_edge)
    ).scalar_one()

    return CorpusHealth(
        missing_doi=missing_doi,
        excluded=excluded,
        claims_without_embeddings=claims_without_embeddings,
        no_citation_coverage=no_citation_coverage,
    )


def _gap_stats(session: Session) -> GapReviewStats:
    """Same shape as _assessment_stats: a review queue's status breakdown,
    grouped straight off CandidateGap.status (see that column's CHECK
    constraint for the three valid values), plus the mean of each Sec 44
    rating dimension across whatever gaps have been rated on it -
    func.avg over a nullable column ignores NULLs on its own, which is
    exactly "mean across gaps rated on this dimension"."""
    counts = dict(
        session.execute(select(CandidateGap.status, func.count()).group_by(CandidateGap.status)).all()
    )
    means = session.execute(
        select(
            func.avg(CandidateGap.correctness_rating),
            func.avg(CandidateGap.relevance_rating),
            func.avg(CandidateGap.novelty_rating),
            func.avg(CandidateGap.evidence_support_rating),
            func.avg(CandidateGap.usefulness_rating),
        )
    ).one()
    return GapReviewStats(
        pending=counts.get("pending", 0),
        approved=counts.get("approved", 0),
        rejected=counts.get("rejected", 0),
        mean_correctness=float(means[0]) if means[0] is not None else None,
        mean_relevance=float(means[1]) if means[1] is not None else None,
        mean_novelty=float(means[2]) if means[2] is not None else None,
        mean_evidence_support=float(means[3]) if means[3] is not None else None,
        mean_usefulness=float(means[4]) if means[4] is not None else None,
    )


ERROR_SAMPLE_LIMIT = 500
"""How many of the most recent ingestion_errors rows _ingestion_errors_by_type
groups over - a sample for spotting what's failing lately, not an
all-time count, so this stays cheap regardless of corpus age."""


def _ingestion_errors_by_type(session: Session) -> dict[str, int]:
    recent_ids = select(IngestionError.id).order_by(IngestionError.occurred_at.desc()).limit(ERROR_SAMPLE_LIMIT)
    rows = session.execute(
        select(IngestionError.error_type, func.count())
        .where(IngestionError.id.in_(recent_ids))
        .group_by(IngestionError.error_type)
    ).all()
    return dict(rows)


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


@router.post("/{key}/stop", response_model=PipelineStopOut)
def stop_pipeline(key: str, session: Session = Depends(get_session)) -> PipelineStopOut:
    """Kill the subprocess running under `key`, if any, and mark its
    in-progress *_runs row "stopped" - the subprocess itself never gets a
    chance to do that since it's killed, not given time to shut down."""
    if key not in PIPELINE_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline key {key!r}")

    stopped = stop(key)
    if not stopped:
        raise HTTPException(status_code=409, detail=f"{key} is not running")

    model_entry = RUN_MODEL_BY_KEY.get(key)
    if model_entry is None:
        return PipelineStopOut(stopped=True, pipeline=key)

    model, source = model_entry
    query = select(model).where(model.status == "running")
    if source is not None:
        query = query.where(model.source == source)
    run = session.execute(query.order_by(model.started_at.desc()).limit(1)).scalar_one_or_none()
    if run is not None:
        run.status = "stopped"
        run.finished_at = datetime.now(timezone.utc)
        run.error_summary = "Stopped by operator"
        session.commit()

    return PipelineStopOut(stopped=True, pipeline=key)


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


@router.post("/ingestion/core/run", response_model=PipelineTriggerOut)
def trigger_core_ingestion(payload: CoreIngestionTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409("ingestion_core", "researchbridge.ingestion.cli_core", args)


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


@router.post("/retrieval-eval/run", response_model=PipelineTriggerOut)
def trigger_retrieval_eval(payload: RetrievalEvalTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.k is not None:
        args += ["--k", str(payload.k)]
    return _trigger_or_409("retrieval_eval", "researchbridge.retrieval.cli_evaluate", args)


@router.get("/retrieval-eval", response_model=RetrievalEvalOut)
def get_retrieval_eval() -> RetrievalEvalOut:
    """Reads the last rb-retrieval-evaluate run's persisted results (see
    RETRIEVAL_EVAL_RESULTS_PATH) - never computes them live, since a real
    run loads an embedding model and fits four retrievers against the
    whole corpus."""
    if not RETRIEVAL_EVAL_RESULTS_PATH.exists():
        return RetrievalEvalOut(available=False, generated_at=None, k=None, query_sets=None)

    data = json.loads(RETRIEVAL_EVAL_RESULTS_PATH.read_text())
    return RetrievalEvalOut(
        available=True,
        generated_at=data["generated_at"],
        k=data["k"],
        query_sets=data["query_sets"],
    )


@router.post("/extraction-eval/run", response_model=PipelineTriggerOut)
def trigger_extraction_eval(payload: ExtractionEvalTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.threshold is not None:
        args += ["--threshold", str(payload.threshold)]
    if payload.extractor is not None:
        args += ["--extractor", payload.extractor]
    return _trigger_or_409("extraction_eval", "researchbridge.extraction.cli_evaluate", args)


@router.get("/extraction-eval", response_model=ExtractionEvalOut)
def get_extraction_eval() -> ExtractionEvalOut:
    """Reads the last rb-extract-evaluate run's persisted results (see
    EXTRACTION_EVAL_RESULTS_PATH) - never computes them live, since a real
    run loads an embedding model and runs one or more extractors against
    the whole benchmark."""
    if not EXTRACTION_EVAL_RESULTS_PATH.exists():
        return ExtractionEvalOut(available=False, generated_at=None, threshold=None, paper_count=None, extractors=None)

    data = json.loads(EXTRACTION_EVAL_RESULTS_PATH.read_text())
    return ExtractionEvalOut(
        available=True,
        generated_at=data["generated_at"],
        threshold=data["threshold"],
        paper_count=data["paper_count"],
        extractors=data["extractors"],
    )


@router.post("/citations-fetch/run", response_model=PipelineTriggerOut)
def trigger_citations_fetch(payload: CitationsFetchTrigger) -> PipelineTriggerOut:
    args: list[str] = ["--all", "--save", "--source", payload.source]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409("citations_fetch", "researchbridge.citations.cli_fetch", args)


@router.get("/citations-fetch", response_model=CitationsFetchOut)
def get_citations_fetch() -> CitationsFetchOut:
    """Reads each source's last rb-citations-fetch --all run's persisted
    summary (see CITATIONS_FETCH_SUMMARY_PATHS) - never triggers a fetch
    here, since a real run walks every eligible paper in the corpus
    against a rate-limited external API."""
    return CitationsFetchOut(
        semantic_scholar=_read_citation_summary(CITATIONS_FETCH_SUMMARY_PATHS["semantic_scholar"]),
        crossref=_read_citation_summary(CITATIONS_FETCH_SUMMARY_PATHS["crossref"]),
    )


def _read_citation_summary(path) -> CitationSourceSummary:
    if not path.exists():
        return CitationSourceSummary(
            available=False, generated_at=None, papers_seen=None, papers_failed=None,
            edges_created=None, edges_already_existed=None,
        )

    data = json.loads(path.read_text())
    return CitationSourceSummary(
        available=True,
        generated_at=data["generated_at"],
        papers_seen=data["papers_seen"],
        papers_failed=data["papers_failed"],
        edges_created=data["edges_created"],
        edges_already_existed=data["edges_already_existed"],
    )
