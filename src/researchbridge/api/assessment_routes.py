"""ResearchInput -> ResearchAssessment endpoints (blueprint Sec 2A).

Two input paths, both synchronous:
- POST /api/assessments: idea text.
- POST /api/assessments/upload: an uploaded document (.pdf or plain text).
  Sec 2A's "Uploaded-Paper Path" - text/section extraction happens inside
  build_assessment() via assessment/representation.py, not here. This
  route's only job is turning the upload into raw_text.

Both call build_assessment(..., enable_llm_stages=ollama_enabled()) -
when Ollama is enabled, applications relevance filtering and opportunity
synthesis (both local-LLM stages, see build.py's own docstring) run
automatically as part of assessment creation; when disabled, both stages
are skipped and the pipeline stays fully synchronous/deterministic/no-
external-dependency, same as before 2026-09-04.

Upload endpoint security review (item 6 of the assessment hardening
list). Fixed, both with real reproductions before and after (see
test_assessment_api.py): (1) no upload size cap at all - `await
file.read()` loaded an arbitrarily large body fully into memory before
any validation ran, a trivial memory-exhaustion DoS; now capped at
MAX_UPLOAD_BYTES via _read_capped's chunked read, rejecting with 413 as
soon as the cap is crossed rather than after buffering the whole thing.
(2) a file merely NAMED ".pdf" but not actually valid PDF bytes crashed
with an unhandled pymupdf.FileDataError -> bare 500, instead of a clean
422 - now caught and reported as a validation error.

Assessed and deliberately NOT changed, to avoid guessing at fixes for
risks that aren't concretely demonstrated here: (a) no auth on this
route - by design (see api/app.py's own docstring: this is a local-only
tool, CORS-restricted to the dev frontend origins only as a browser-
same-origin convenience, not a real access-control boundary against a
direct request) - adding auth would be a project-wide architecture
change, out of scope for one endpoint's review. (b) no per-request rate
limiting - same local-only reasoning; there is no shared/multi-tenant
deployment target yet to rate-limit against. (c) no PDF "decompression
bomb" / adversarial-page-count guard on pymupdf's own parsing - MuPDF is
a mature, actively maintained C library and no concrete exploit or
resource-exhaustion reproduction was found against it here; the
MAX_UPLOAD_BYTES cap already bounds the size of the input pymupdf ever
sees. (d) filename is never used to open/write anything on disk (only
regex-matched against an arxiv-id pattern in assessment/matching.py and
stored as a plain DB string, rendered as JSX text in the frontend, never
via dangerouslySetInnerHTML) - no path-traversal or stored-XSS route
found.
"""

from __future__ import annotations

import uuid

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import and_, case, delete, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    GraphEdgeOut,
    GraphNodeOut,
    ResearchAssessmentCreate,
    ResearchAssessmentHistoryItem,
    ResearchAssessmentOut,
    ResearchAssessmentReview,
    ResearchAssessmentSummaryOut,
    ResearchAssessmentSummaryPage,
    SimilarityGraphOut,
)
from researchbridge.api.serializers import to_assessment_claims, to_assessment_evidence
from researchbridge.assessment.build import build_assessment
from researchbridge.assessment.claims import sync_claim_status
from researchbridge.assessment.export import build_docx, build_markdown, build_pdf
from researchbridge.assessment.graph import build_similarity_graph
from researchbridge.assessment.matching import match_uploaded_paper
from researchbridge.assessment.opportunity_synthesis import (
    OpportunitySynthesisUnavailable,
    SourceApplication,
    ollama_enabled,
    synthesize_opportunities,
    to_persisted_opportunities,
)
from researchbridge.benchmark.fulltext import extract_text
from researchbridge.db.models import ResearchAssessment, ResearchAssessmentEvidence, ResearchInput
from researchbridge.embedding.base import Embedder

router = APIRouter(prefix="/api/assessments")

# Item 6 of the assessment hardening list (security review of the upload
# endpoint): `await file.read()` had no size cap at all - a single upload
# of arbitrary size was read entirely into memory before any validation
# ran, a trivial memory-exhaustion DoS on this local-only, unauthenticated
# endpoint (see api/app.py's own docstring: CORS-restricted to the dev
# frontend, but that is not an auth boundary against a direct request).
# 25MB is generous for a real paper PDF/text file (the corpus's actual PDFs
# run a few MB) while still bounding worst-case memory use per upload to a
# known, small constant.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024

MAX_LIST_LIMIT = 100
_REVIEW_FILTERS = {"all", "reviewed", "needs_review"}
_SORT_OPTIONS = {"newest", "priority"}
_INPUT_PREVIEW_LENGTH = 120

# Order-of-preference for "priority" sort, lowest rank first. Not a fabricated
# score (blueprint Sec 18/42) - this only reorders by the recommendation the
# pipeline already computed, same categorical value shown elsewhere in the
# app. A NULL recommendation (assessment not yet completed) sorts last.
_PRIORITY_RANK = {
    "HIGH PRIORITY": 0,
    "MEDIUM PRIORITY": 1,
    "LOW PRIORITY": 2,
    "REQUIRES HUMAN REVIEW": 3,
    "INSUFFICIENT EVIDENCE": 4,
}


@router.get("", response_model=ResearchAssessmentSummaryPage)
def list_assessments(
    session: Session = Depends(get_session),
    review: str = Query("all", description="all, reviewed, or needs_review"),
    sort: str = Query("newest", description="newest or priority"),
    novelty: str | None = Query(None, description="Filter by novelty_level: high, medium, low, or not_assessed"),
    feasibility: str | None = Query(
        None, description="Filter by technical_feasibility_level: high, medium, low, or not_assessed"
    ),
    limit: int = Query(20, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(0, ge=0),
) -> ResearchAssessmentSummaryPage:
    """The latest assessment per research_input - re-run history collapses to
    one row (see the assessment's own /history for the rest). "priority" sort
    orders by the pipeline's own recommendation tier (see _PRIORITY_RANK) -
    not a new score, just a defensible ordering over an existing field."""
    if review not in _REVIEW_FILTERS:
        raise HTTPException(status_code=422, detail=f"review must be one of {sorted(_REVIEW_FILTERS)}")
    if sort not in _SORT_OPTIONS:
        raise HTTPException(status_code=422, detail=f"sort must be one of {sorted(_SORT_OPTIONS)}")

    latest = (
        select(
            ResearchAssessment.research_input_id,
            func.max(ResearchAssessment.created_at).label("max_created_at"),
        )
        .group_by(ResearchAssessment.research_input_id)
        .subquery()
    )

    query = (
        select(ResearchAssessment, ResearchInput)
        .join(ResearchInput, ResearchInput.id == ResearchAssessment.research_input_id)
        .join(
            latest,
            and_(
                ResearchAssessment.research_input_id == latest.c.research_input_id,
                ResearchAssessment.created_at == latest.c.max_created_at,
            ),
        )
    )
    if review == "reviewed":
        query = query.where(ResearchAssessment.human_reviewed.is_(True))
    elif review == "needs_review":
        query = query.where(ResearchAssessment.human_reviewed.is_(False))
    if novelty is not None:
        query = query.where(ResearchAssessment.novelty_level == novelty)
    if feasibility is not None:
        query = query.where(ResearchAssessment.technical_feasibility_level == feasibility)

    if sort == "priority":
        priority_rank = case(_PRIORITY_RANK, value=ResearchAssessment.recommendation, else_=len(_PRIORITY_RANK))
        ordering = (priority_rank, ResearchAssessment.created_at.desc())
    else:
        ordering = (ResearchAssessment.created_at.desc(),)

    total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()
    rows = session.execute(query.order_by(*ordering).limit(limit).offset(offset)).all()

    return ResearchAssessmentSummaryPage(
        items=[_to_summary(assessment, research_input) for assessment, research_input in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


def _to_summary(assessment: ResearchAssessment, research_input: ResearchInput) -> ResearchAssessmentSummaryOut:
    preview = (
        research_input.source_filename
        if research_input.input_type == "document" and research_input.source_filename
        else research_input.raw_text
    )
    if len(preview) > _INPUT_PREVIEW_LENGTH:
        preview = preview[:_INPUT_PREVIEW_LENGTH] + "…"

    return ResearchAssessmentSummaryOut(
        id=assessment.id,
        created_at=assessment.created_at,
        status=assessment.status,
        novelty_level=assessment.novelty_level,
        technical_feasibility_level=assessment.technical_feasibility_level,
        recommendation=assessment.recommendation,
        confidence=assessment.confidence,
        human_reviewed=assessment.human_reviewed,
        research_input_id=research_input.id,
        input_type=research_input.input_type,
        input_preview=preview,
    )


@router.post("", response_model=ResearchAssessmentOut)
def create_assessment(
    payload: ResearchAssessmentCreate,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> ResearchAssessmentOut:
    research_input = ResearchInput(input_type="idea", raw_text=payload.raw_text)
    session.add(research_input)
    session.flush()

    assessment = build_assessment(session, research_input.id, embedder, enable_llm_stages=ollama_enabled())

    return _to_out(session, assessment, research_input)


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Reads `file` in fixed-size chunks, rejecting with 413 as soon as
    max_bytes is exceeded - unlike a bare `await file.read()`, this never
    holds more than one chunk beyond the cap in memory regardless of how
    large the actual upload is."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"uploaded file exceeds the {max_bytes // (1024 * 1024)}MB limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload", response_model=ResearchAssessmentOut)
async def create_assessment_from_upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> ResearchAssessmentOut:
    content = await _read_capped(file, MAX_UPLOAD_BYTES)
    if not content:
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    filename = file.filename or ""
    if filename.lower().endswith(".pdf"):
        try:
            text = extract_text(content)
        except pymupdf.FileDataError:
            raise HTTPException(status_code=422, detail="uploaded file is not a valid PDF") from None
    else:
        text = content.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(status_code=422, detail="no extractable text found in the uploaded file")

    research_input = ResearchInput(
        input_type="document",
        raw_text=text,
        source_filename=filename or None,
        matched_paper_id=match_uploaded_paper(session, filename or None, text),
    )
    session.add(research_input)
    session.flush()

    assessment = build_assessment(session, research_input.id, embedder, enable_llm_stages=ollama_enabled())

    return _to_out(session, assessment, research_input)


@router.get("/{assessment_id}", response_model=ResearchAssessmentOut)
def get_assessment(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> ResearchAssessmentOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


@router.get("/{assessment_id}/graph", response_model=SimilarityGraphOut)
def get_assessment_graph(
    assessment_id: uuid.UUID,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> SimilarityGraphOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    graph = build_similarity_graph(session, assessment, research_input, embedder)

    return SimilarityGraphOut(
        nodes=[
            GraphNodeOut(
                id=node.id, type=node.type, title=node.title,
                distance_to_input=node.distance_to_input, claim_counts=node.claim_counts,
            )
            for node in graph.nodes
        ],
        edges=[GraphEdgeOut(source=edge.source, target=edge.target, distance=edge.distance) for edge in graph.edges],
    )


@router.delete("/{assessment_id}", status_code=204)
def delete_assessment(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> None:
    """Deletes the whole assessment thread - every rerun for the same
    research_input (see rerun_assessment: a rerun always adds a new row,
    never overwrites), not just the one row `assessment_id` names.

    The list page collapses to "latest per research_input" (see
    list_assessments), so deleting only this one row would just let the
    next-most-recent rerun silently resurface as "the" entry - not what
    "delete this from the list" means from that page.
    """
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input_id = assessment.research_input_id
    thread_ids = select(ResearchAssessment.id).where(ResearchAssessment.research_input_id == research_input_id)
    session.execute(
        delete(ResearchAssessmentEvidence).where(ResearchAssessmentEvidence.research_assessment_id.in_(thread_ids))
    )
    session.execute(delete(ResearchAssessment).where(ResearchAssessment.research_input_id == research_input_id))
    session.execute(delete(ResearchInput).where(ResearchInput.id == research_input_id))
    session.commit()


@router.put("/{assessment_id}/review", response_model=ResearchAssessmentOut)
def review_assessment(
    assessment_id: uuid.UUID, payload: ResearchAssessmentReview, session: Session = Depends(get_session)
) -> ResearchAssessmentOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    assessment.human_reviewed = payload.human_reviewed
    sync_claim_status(session, assessment)
    session.commit()

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


@router.post("/{assessment_id}/rerun", response_model=ResearchAssessmentOut)
def rerun_assessment(
    assessment_id: uuid.UUID,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> ResearchAssessmentOut:
    """Re-run the pipeline for an existing assessment's research_input.

    Always creates a NEW ResearchAssessment row rather than overwriting the
    original - the corpus grows over time (new ingestion, new extraction),
    so a re-run's results can genuinely differ, and preserving both keeps
    the assessment history honest rather than silently replacing it."""
    original = session.get(ResearchAssessment, assessment_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    assessment = build_assessment(session, original.research_input_id, embedder, enable_llm_stages=ollama_enabled())

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


@router.post("/{assessment_id}/opportunities", response_model=ResearchAssessmentOut)
def synthesize_assessment_opportunities(
    assessment_id: uuid.UUID, session: Session = Depends(get_session)
) -> ResearchAssessmentOut:
    """Synthesizes and persists Direct/Adjacent/Speculative product
    opportunities from this assessment's own potential_applications - see
    docs/superpowers/specs/2026-09-03-opportunities-synthesis-design.md.
    Unlike every other assess_* field, this one is generated on demand
    (not during build_assessment()) by a local LLM: 422 if there are no
    applications to ground it in, 503 if OLLAMA_ENABLED is false or the
    model can't produce a validly-cited result after one retry - never a
    partial result persisted."""
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")
    if not assessment.potential_applications:
        raise HTTPException(status_code=422, detail="no potential applications to ground opportunity synthesis in")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    applications = [
        SourceApplication(
            application=a["application"],
            source_paper=a["source_paper"],
            paper_id=a["paper_id"],
            evidence_id=a.get("evidence_id"),
        )
        for a in assessment.potential_applications
    ]

    try:
        result = synthesize_opportunities(research_input.raw_text, applications)
    except OpportunitySynthesisUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    assessment.potential_opportunities, cited_evidence_ids = to_persisted_opportunities(applications, result)

    # Link the same real evidence the cited applications already trace to -
    # never fabricated for this field. This route can be called again on
    # the same assessment (e.g. to regenerate after a failed/unsatisfying
    # attempt), so any evidence links from a previous synthesis are cleared
    # first - otherwise a re-run citing the same application a second time
    # would violate the unique constraint on (assessment_id, evidence_id,
    # role) instead of just replacing the old link.
    session.execute(
        delete(ResearchAssessmentEvidence).where(
            ResearchAssessmentEvidence.research_assessment_id == assessment.id,
            ResearchAssessmentEvidence.role == "opportunity",
        )
    )
    for evidence_id in cited_evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=uuid.UUID(evidence_id), role="opportunity"
            )
        )

    session.commit()

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


@router.get("/{assessment_id}/history", response_model=list[ResearchAssessmentHistoryItem])
def assessment_history(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> list[ResearchAssessment]:
    """All assessments (including this one) for the same research_input, newest first."""
    original = session.get(ResearchAssessment, assessment_id)
    if original is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    return list(
        session.execute(
            select(ResearchAssessment)
            .where(ResearchAssessment.research_input_id == original.research_input_id)
            .order_by(ResearchAssessment.created_at.desc())
        ).scalars()
    )


@router.get("/{assessment_id}/export.docx")
def export_assessment_docx(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    out = _load_for_export(session, assessment_id)
    return Response(
        content=build_docx(out),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=assessment-{assessment_id}.docx"},
    )


@router.get("/{assessment_id}/export.pdf")
def export_assessment_pdf(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    out = _load_for_export(session, assessment_id)
    return Response(
        content=build_pdf(out),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=assessment-{assessment_id}.pdf"},
    )


@router.get("/{assessment_id}/export.md")
def export_assessment_markdown(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    out = _load_for_export(session, assessment_id)
    return Response(
        content=build_markdown(out),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=assessment-{assessment_id}.md"},
    )


def _load_for_export(session: Session, assessment_id: uuid.UUID) -> ResearchAssessmentOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


def _to_out(
    session: Session, assessment: ResearchAssessment, research_input: ResearchInput
) -> ResearchAssessmentOut:
    return ResearchAssessmentOut(
        id=assessment.id,
        created_at=assessment.created_at,
        research_input=research_input,
        status=assessment.status,
        retrieved_paper_ids=assessment.retrieved_paper_ids,
        comparison_summary=assessment.comparison_summary,
        novelty_level=assessment.novelty_level,
        novelty_reasoning=assessment.novelty_reasoning,
        research_gap_text=assessment.research_gap_text,
        research_gap_source=assessment.research_gap_source,
        candidate_gap_id=assessment.candidate_gap_id,
        potential_applications=assessment.potential_applications,
        technical_feasibility_level=assessment.technical_feasibility_level,
        technical_feasibility_reasoning=assessment.technical_feasibility_reasoning,
        potential_opportunities=assessment.potential_opportunities,
        risks_and_limitations=assessment.risks_and_limitations,
        external_validation_needed=assessment.external_validation_needed,
        recommendation=assessment.recommendation,
        confidence=assessment.confidence,
        human_reviewed=assessment.human_reviewed,
        evidence=to_assessment_evidence(session, assessment.id),
        claims=to_assessment_claims(session, assessment.id),
    )
