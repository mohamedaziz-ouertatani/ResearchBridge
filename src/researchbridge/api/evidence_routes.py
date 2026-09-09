"""Human review feedback for grounded evidence passages."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.schemas import EvidenceReviewCreate, EvidenceReviewOut
from researchbridge.db.models import (
    Evidence,
    EvidenceReview,
    QaQuestion,
    ResearchAssessment,
    ResearchAssessmentEvidence,
)

router = APIRouter(prefix="/api/evidence")


@router.put("/review", response_model=EvidenceReviewOut)
def review_evidence(payload: EvidenceReviewCreate, session: Session = Depends(get_session)) -> EvidenceReviewOut:
    if session.get(Evidence, payload.evidence_id) is None:
        raise HTTPException(status_code=404, detail=f"No evidence with id {payload.evidence_id}")

    if payload.surface_type == "assessment":
        if session.get(ResearchAssessment, payload.surface_id) is None:
            raise HTTPException(status_code=404, detail=f"No assessment with id {payload.surface_id}")
        linked = session.execute(
            select(ResearchAssessmentEvidence.id).where(
                ResearchAssessmentEvidence.research_assessment_id == payload.surface_id,
                ResearchAssessmentEvidence.evidence_id == payload.evidence_id,
                ResearchAssessmentEvidence.role == payload.role,
            )
        ).scalar_one_or_none()
        if linked is None:
            raise HTTPException(status_code=422, detail="evidence is not linked to this assessment role")
    elif session.get(QaQuestion, payload.surface_id) is None:
        raise HTTPException(status_code=404, detail=f"No saved Q&A question with id {payload.surface_id}")

    review = session.execute(
        select(EvidenceReview).where(
            EvidenceReview.surface_type == payload.surface_type,
            EvidenceReview.surface_id == payload.surface_id,
            EvidenceReview.evidence_id == payload.evidence_id,
            EvidenceReview.role == payload.role,
        )
    ).scalar_one_or_none()
    if review is None:
        review = EvidenceReview(
            evidence_id=payload.evidence_id,
            surface_type=payload.surface_type,
            surface_id=payload.surface_id,
            role=payload.role,
            verdict=payload.verdict,
            note=payload.note,
        )
        session.add(review)
    else:
        review.verdict = payload.verdict
        review.note = payload.note
        review.updated_at = datetime.now(UTC)
    session.commit()
    return review
