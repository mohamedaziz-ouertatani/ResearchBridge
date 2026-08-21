"""ResearchInput -> ResearchAssessment endpoints (blueprint Sec 2A).

First vertical slice: idea-only input, synchronous (the whole pipeline is
local/fast - retrieval + reading already-extracted claims, no new model
calls). Uploaded-document input and richer report fields (novelty, gap,
feasibility, recommendation) are later enrichment passes, not this slice.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import ResearchAssessmentCreate, ResearchAssessmentOut
from researchbridge.assessment.build import build_assessment
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

    return _to_out(assessment, research_input)


@router.get("/{assessment_id}", response_model=ResearchAssessmentOut)
def get_assessment(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> ResearchAssessmentOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    return _to_out(assessment, research_input)


def _to_out(assessment: ResearchAssessment, research_input: ResearchInput) -> ResearchAssessmentOut:
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
        technical_feasibility_level=assessment.technical_feasibility_level,
        recommendation=assessment.recommendation,
        confidence=assessment.confidence,
    )
