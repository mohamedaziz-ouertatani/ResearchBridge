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

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    ResearchAssessmentCreate,
    ResearchAssessmentHistoryItem,
    ResearchAssessmentOut,
    ResearchAssessmentReview,
)
from researchbridge.api.serializers import to_assessment_evidence
from researchbridge.assessment.build import build_assessment
from researchbridge.assessment.export import build_docx, build_pdf
from researchbridge.assessment.matching import match_uploaded_paper
from researchbridge.benchmark.fulltext import extract_text
from researchbridge.db.models import ResearchAssessment, ResearchInput
from researchbridge.embedding.base import Embedder

router = APIRouter(prefix="/api/assessments")


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
