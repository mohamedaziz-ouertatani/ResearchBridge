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

import psutil
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, exists, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.pipeline_triggers import PipelineAlreadyRunning, is_running, stop, tail_log, trigger
from researchbridge.api.schemas import (
    ArxivIngestionTrigger,
    AssessmentStats,
    CitationsFetchTrigger,
    CoreIngestionTrigger,
    CorpusHealth,
    EmbeddingTrigger,
    ExtractionEvalOut,
    ExtractionEvalTrigger,
    ExtractionTrigger,
    FullTextFetchTrigger,
    GapReviewStats,
    Notification,
    PaperExclude,
    PaperFullTextOut,
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
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    CitationFetchRun,
    Embedding,
    EmbeddingRun,
    ExtractedClaim,
    ExtractionRun,
    FullTextFetchRun,
    GapDetectionRun,
    IngestionError,
    IngestionRun,
    Paper,
    PaperCitation,
    PaperFullText,
    ResearchAssessment,
)
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
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
    "gaps",
    "fulltext",
)
""""gaps" is gaps_routes.py's PIPELINE_KEY, not one this router's own trigger
endpoints start - POST /api/gaps/detect (--all --save, no params) is the
only way to launch it. It's included here anyway so the generic /{key}/log
and /{key}/stop endpoints, and the `running` flag below, work for it the
same as every admin-triggered pipeline - one shared subprocess registry
(pipeline_triggers.py), one status shape, regardless of which router
started the subprocess."""

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
    "gaps": (GapDetectionRun, None),
    "fulltext": (FullTextFetchRun, None),
}


@router.get("/pipeline", response_model=PipelineStatus)
def pipeline_status(session: Session = Depends(get_session)) -> PipelineStatus:
    _reconcile_stale_running_runs(session)

    total_papers = session.execute(select(func.count(Paper.id))).scalar_one()
    papers_with_claims = session.execute(
        select(func.count(func.distinct(ExtractedClaim.paper_id)))
    ).scalar_one()
    papers_with_embeddings = session.execute(
        select(func.count(func.distinct(Embedding.paper_id)))
    ).scalar_one()
    papers_with_fulltext = session.execute(
        select(func.count(func.distinct(PaperFullText.paper_id)))
    ).scalar_one()
    papers_by_source = dict(
        session.execute(select(Paper.source, func.count(Paper.id)).group_by(Paper.source)).all()
    )

    return PipelineStatus(
        total_papers=total_papers,
        papers_with_claims=papers_with_claims,
        papers_with_embeddings=papers_with_embeddings,
        papers_with_fulltext=papers_with_fulltext,
        papers_by_source=papers_by_source,
        corpus_health=_corpus_health(session),
        assessment_stats=_assessment_stats(session),
        gap_stats=_gap_stats(session),
        ingestion_errors_by_type=_ingestion_errors_by_type(session),
        analysis_claims_by_type=_analysis_claims_by_type(session),
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
        gap_detection_runs=[
            _to_run(
                run,
                (
                    "papers_seen",
                    "papers_skipped",
                    "papers_failed",
                    "no_relevant_papers",
                    "insufficient_evidence",
                    "gaps_found",
                    "gaps_saved",
                ),
            )
            for run in _recent(session, GapDetectionRun)
        ],
        fulltext_fetch_runs=[
            _to_run(run, ("papers_seen", "papers_fetched", "papers_skipped_no_url", "papers_failed"))
            for run in _recent(session, FullTextFetchRun)
        ],
        running={key: is_running(key) or has_running_db_row(session, key) for key in PIPELINE_KEYS},
    )


def has_running_db_row(session: Session, key: str) -> bool:
    """A pipeline key counts as running if its own *_runs table has a row
    in progress AND that row's own process is actually still alive, even
    when this API server has no live subprocess handle for it - covers a
    run started directly from the CLI (bypassing the trigger button
    entirely) or one that outlived a server restart, neither of which
    is_running()'s in-process registry can see (see pipeline_triggers.py's
    module docstring).

    status="running" alone is NOT enough: a process that crashed or was
    killed without reaching the code that marks its row "failed"/"stopped"
    leaves a row stuck saying "running" forever, with nothing behind it -
    checking pid liveness (psutil.pid_exists) is what tells that apart
    from a run that's genuinely still going. A row with no pid (written
    before this column existed) can't be verified either way, so it
    doesn't count as running - see migration 0019's docstring for why
    that's the safer default than assuming True.

    Keys with no *_runs table (retrieval_eval, extraction_eval - one-off
    diagnostics, see RUN_MODEL_BY_KEY's docstring) always return False
    here; is_running() alone is the whole story for those.
    """
    entry = RUN_MODEL_BY_KEY.get(key)
    if entry is None:
        return False
    model, source = entry
    conditions = [model.status == "running"]
    if source is not None:
        conditions.append(model.source == source)
    pids = session.execute(select(model.pid).where(and_(*conditions))).scalars().all()
    return any(pid is not None and psutil.pid_exists(pid) for pid in pids)


def _reconcile_stale_running_runs(session: Session) -> None:
    """Self-heals a *_runs row stuck at status="running" whose process is
    no longer alive - e.g. a run started directly from the CLI (or one
    killed outside this API, like Stop-Process) never reaches the code
    that would otherwise mark its row "failed"/"stopped", leaving it
    reading "running" forever even after has_running_db_row() (and this
    endpoint's `running` flag) correctly stop counting it as live.

    Runs on every call to this endpoint - which the admin panel already
    polls every 15s - so a stuck row corrects itself within one poll
    cycle instead of persisting until someone notices and fixes it by
    hand. A row with no pid (written before that column existed) can't be
    verified either way, so it's left alone, same as has_running_db_row().
    """
    changed = False
    for model, source in RUN_MODEL_BY_KEY.values():
        conditions = [model.status == "running"]
        if source is not None:
            conditions.append(model.source == source)
        rows = session.execute(select(model).where(and_(*conditions))).scalars().all()
        for row in rows:
            if row.pid is not None and not psutil.pid_exists(row.pid):
                row.status = "failed"
                row.error_summary = "Process no longer running - marked failed automatically."
                row.finished_at = datetime.now(timezone.utc)
                changed = True
    if changed:
        session.commit()


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

    for run in _recent_finished(session, CitationFetchRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type=f"citations_fetch_{run.status}",
                severity=_severity(run.status),
                message=_citations_fetch_message(run),
                created_at=run.finished_at or run.started_at,
            )
        )

    for run in _recent_finished(session, GapDetectionRun):
        items.append(
            Notification(
                id=f"run:{run.id}",
                type=f"gap_detection_{run.status}",
                severity=_severity(run.status),
                message=_gap_detection_message(run),
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


def _citations_fetch_message(run: CitationFetchRun) -> str:
    label = run.source.replace("_", " ")
    if run.status == "failed":
        return f"{label} citation fetch failed: {run.error_summary or 'unknown error'}"
    if run.status == "stopped":
        return f"{label} citation fetch stopped: {run.edges_created} edges created so far"
    return (
        f"{label} citation fetch completed: {run.edges_created} edges created, "
        f"{run.edges_already_existed} already existed, {run.papers_failed} failed"
    )


def _gap_detection_message(run: GapDetectionRun) -> str:
    forced = " (forced)" if run.force else ""
    if run.status == "failed":
        return f"Gap detection run failed{forced}: {run.error_summary or 'unknown error'}"
    if run.status == "stopped":
        return f"Gap detection run stopped{forced}: {run.gaps_saved} gap(s) saved so far"
    return f"Gap detection run completed{forced}: {run.gaps_saved} gap(s) saved, {run.papers_failed} paper(s) failed"


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

    is_embedded = exists().where(Embedding.paper_id == Paper.id, Embedding.embedding_type == EMBEDDING_TYPE)
    has_candidate_gap = exists().where(CandidateGap.seed_paper_id == Paper.id)
    not_gap_processed = session.execute(
        select(func.count())
        .select_from(Paper)
        .where(is_embedded, Paper.excluded_at.is_(None), ~has_candidate_gap)
    ).scalar_one()

    return CorpusHealth(
        missing_doi=missing_doi,
        excluded=excluded,
        claims_without_embeddings=claims_without_embeddings,
        no_citation_coverage=no_citation_coverage,
        not_gap_processed=not_gap_processed,
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


def _analysis_claims_by_type(session: Session) -> dict[str, int]:
    """Corpus-wide count of every analysis_claims row by claim_type (fact/
    inference/hypothesis/opportunity/speculation), across both producers
    (gaps/claims.py, assessment/claims.py) - an all-time count, not a
    recent sample, since this table only ever grows one row per already-
    reviewed-worthy gap/assessment field, nothing like ingestion's
    error-log volume."""
    return dict(
        session.execute(select(AnalysisClaim.claim_type, func.count()).group_by(AnalysisClaim.claim_type)).all()
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
        # Only meaningful while a row is still "running" - once it's
        # finished/failed/stopped, the process behind it is gone (or
        # reused for something else entirely), so showing a stale pid
        # would be noise, not a useful correlation.
        pid=run.pid if run.status == "running" else None,
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

    model_entry = RUN_MODEL_BY_KEY.get(key)
    run = None
    if model_entry is not None:
        model, source = model_entry
        query = select(model).where(model.status == "running")
        if source is not None:
            query = query.where(model.source == source)
        run = session.execute(query.order_by(model.started_at.desc()).limit(1)).scalar_one_or_none()

    # fallback_pid (2026-09-05): the in-memory subprocess registry resets on
    # server restart, but the subprocess itself is deliberately detached so
    # it survives that restart - without this, a run that outlived a
    # restart could never be stopped again, 409ing forever against a
    # process that's genuinely still alive. See pipeline_triggers.stop()'s
    # own docstring for the full reasoning; same pid this endpoint's own
    # has_running_db_row()/_reconcile_stale_running_runs already trust for
    # liveness detection.
    stopped = stop(key, fallback_pid=run.pid if run is not None else None)
    if not stopped:
        raise HTTPException(status_code=409, detail=f"{key} is not running")

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


@router.get("/papers/{paper_id}/fulltext", response_model=PaperFullTextOut)
def get_paper_fulltext(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> PaperFullTextOut:
    fulltext = session.execute(
        select(PaperFullText).where(PaperFullText.paper_id == paper_id)
    ).scalar_one_or_none()
    if fulltext is None:
        raise HTTPException(status_code=404, detail=f"No full text stored for paper {paper_id}")

    return PaperFullTextOut.model_validate(fulltext)


def _trigger_or_409(session: Session, key: str, module: str, args: list[str]) -> PipelineTriggerOut:
    """Refuses to start a second run for `key` when one is already alive -
    checking both is_running()'s in-process registry (a run this server
    itself spawned) AND has_running_db_row()'s pid-verified DB row (a run
    started directly from the CLI, which the registry alone can't see).
    Without the second check, a CLI-started run - invisible to is_running()
    - would let a panel click launch a genuine concurrent duplicate against
    the same pipeline key, exactly the race this guard exists to prevent.
    """
    if is_running(key) or has_running_db_row(session, key):
        raise HTTPException(status_code=409, detail=f"{key} is already running")
    try:
        log_path = trigger(key, module, args)
    except PipelineAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PipelineTriggerOut(started=True, pipeline=key, log_file=str(log_path))


@router.post("/ingestion/arxiv/run", response_model=PipelineTriggerOut)
def trigger_arxiv_ingestion(
    payload: ArxivIngestionTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.search_query is not None:
        args += ["--search-query", payload.search_query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409(session, "ingestion_arxiv", "researchbridge.ingestion.cli", args)


@router.post("/ingestion/springer/run", response_model=PipelineTriggerOut)
def trigger_springer_ingestion(
    payload: SpringerIngestionTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409(session, "ingestion_springer", "researchbridge.ingestion.cli_springer", args)


@router.post("/ingestion/semantic-scholar/run", response_model=PipelineTriggerOut)
def trigger_semantic_scholar_ingestion(
    payload: SemanticScholarIngestionTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409(
        session, "ingestion_semantic_scholar", "researchbridge.ingestion.cli_semantic_scholar", args
    )


@router.post("/ingestion/core/run", response_model=PipelineTriggerOut)
def trigger_core_ingestion(
    payload: CoreIngestionTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.query is not None:
        args += ["--query", payload.query]
    if payload.page_size is not None:
        args += ["--page-size", str(payload.page_size)]
    if payload.max_pages is not None:
        args += ["--max-pages", str(payload.max_pages)]
    return _trigger_or_409(session, "ingestion_core", "researchbridge.ingestion.cli_core", args)


@router.post("/extraction/run", response_model=PipelineTriggerOut)
def trigger_extraction(payload: ExtractionTrigger, session: Session = Depends(get_session)) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.extractor is not None:
        args += ["--extractor", payload.extractor]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409(session, "extraction", "researchbridge.extraction.cli", args)


@router.post("/embedding/run", response_model=PipelineTriggerOut)
def trigger_embedding(payload: EmbeddingTrigger, session: Session = Depends(get_session)) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409(session, "embedding", "researchbridge.embedding.cli_embed", args)


@router.post("/fulltext/run", response_model=PipelineTriggerOut)
def trigger_fulltext(
    payload: FullTextFetchTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409(session, "fulltext", "researchbridge.fulltext.cli", args)


@router.post("/retrieval-eval/run", response_model=PipelineTriggerOut)
def trigger_retrieval_eval(
    payload: RetrievalEvalTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.k is not None:
        args += ["--k", str(payload.k)]
    return _trigger_or_409(session, "retrieval_eval", "researchbridge.retrieval.cli_evaluate", args)


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
def trigger_extraction_eval(
    payload: ExtractionEvalTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.threshold is not None:
        args += ["--threshold", str(payload.threshold)]
    if payload.extractor is not None:
        args += ["--extractor", payload.extractor]
    return _trigger_or_409(session, "extraction_eval", "researchbridge.extraction.cli_evaluate", args)


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
def trigger_citations_fetch(
    payload: CitationsFetchTrigger, session: Session = Depends(get_session)
) -> PipelineTriggerOut:
    args: list[str] = ["--all", "--save", "--source", payload.source]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409(session, "citations_fetch", "researchbridge.citations.cli_fetch", args)
