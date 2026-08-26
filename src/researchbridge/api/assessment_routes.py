"""ResearchInput -> ResearchAssessment endpoints (blueprint Sec 2A).

Two input paths, both synchronous (the whole pipeline is local/fast -
retrieval + reading already-extracted claims, no new model calls):
- POST /api/assessments: idea text.
- POST /api/assessments/upload: an uploaded document (.pdf or plain text).
  Sec 2A's "Uploaded-Paper Path" - text/section extraction happens inside
  build_assessment() via assessment/representation.py, not here. This
  route's only job is turning the upload into raw_text.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import and_, delete, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    ResearchAssessmentCreate,
    ResearchAssessmentHistoryItem,
    ResearchAssessmentOut,
    ResearchAssessmentReview,
    ResearchAssessmentSummaryOut,
    ResearchAssessmentSummaryPage,
)
from researchbridge.api.serializers import to_assessment_evidence
from researchbridge.assessment.build import build_assessment
from researchbridge.assessment.export import build_docx, build_pdf
from researchbridge.assessment.matching import match_uploaded_paper
from researchbridge.benchmark.fulltext import extract_text
from researchbridge.db.models import ResearchAssessment, ResearchAssessmentEvidence, ResearchInput
from researchbridge.embedding.base import Embedder

router = APIRouter(prefix="/api/assessments")

MAX_LIST_LIMIT = 100
_REVIEW_FILTERS = {"all", "reviewed", "needs_review"}
_INPUT_PREVIEW_LENGTH = 120


@router.get("", response_model=ResearchAssessmentSummaryPage)
def list_assessments(
    session: Session = Depends(get_session),
    review: str = Query("all", description="all, reviewed, or needs_review"),
    limit: int = Query(20, ge=1, le=MAX_LIST_LIMIT),
    offset: int = Query(0, ge=0),
) -> ResearchAssessmentSummaryPage:
    """The latest assessment per research_input, newest first - re-run history
    collapses to one row (see the assessment's own /history for the rest)."""
    if review not in _REVIEW_FILTERS:
        raise HTTPException(status_code=422, detail=f"review must be one of {sorted(_REVIEW_FILTERS)}")

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

    total = session.execute(select(func.count()).select_from(query.subquery())).scalar_one()
    rows = session.execute(
        query.order_by(ResearchAssessment.created_at.desc()).limit(limit).offset(offset)
    ).all()

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

    assessment = build_assessment(session, research_input.id, embedder)

    return _to_out(session, assessment, research_input)


@router.post("/upload", response_model=ResearchAssessmentOut)
async def create_assessment_from_upload(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> ResearchAssessmentOut:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="uploaded file is empty")

    filename = file.filename or ""
    text = extract_text(content) if filename.lower().endswith(".pdf") else content.decode("utf-8", errors="replace")
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

    assessment = build_assessment(session, research_input.id, embedder)

    return _to_out(session, assessment, research_input)


@router.get("/{assessment_id}", response_model=ResearchAssessmentOut)
def get_assessment(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> ResearchAssessmentOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(session, assessment, research_input)


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

    assessment = build_assessment(session, original.research_input_id, embedder)

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
    )
