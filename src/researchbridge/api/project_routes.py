"""Research project workspace endpoints.

Projects are local, unauthenticated groupings of stable ResearchInput rows.
Membership follows the input rather than one rerun so the workspace always
shows the latest assessment for that research idea.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.schemas import (
    ResearchAssessmentSummaryOut,
    ResearchProjectCreate,
    ResearchProjectOut,
    ResearchProjectSummaryOut,
    ResearchProjectUpdate,
)
from researchbridge.db.models import ResearchAssessment, ResearchInput, ResearchProject, ResearchProjectInput

router = APIRouter(prefix="/api/projects")


def _project_or_404(session: Session, project_id: uuid.UUID) -> ResearchProject:
    project = session.get(ResearchProject, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"No research project with id {project_id}")
    return project


def _assessment_summary(assessment: ResearchAssessment, research_input: ResearchInput) -> ResearchAssessmentSummaryOut:
    preview = (
        research_input.source_filename
        if research_input.input_type == "document" and research_input.source_filename
        else research_input.raw_text
    )
    if len(preview) > 240:
        preview = preview[:240] + "…"
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


def _latest_assessments(session: Session, input_ids: list[uuid.UUID]) -> list[ResearchAssessmentSummaryOut]:
    if not input_ids:
        return []
    latest = (
        select(
            ResearchAssessment.research_input_id,
            func.max(ResearchAssessment.created_at).label("max_created_at"),
        )
        .where(ResearchAssessment.research_input_id.in_(input_ids))
        .group_by(ResearchAssessment.research_input_id)
        .subquery()
    )
    rows = session.execute(
        select(ResearchAssessment, ResearchInput)
        .join(ResearchInput, ResearchInput.id == ResearchAssessment.research_input_id)
        .join(
            latest,
            and_(
                ResearchAssessment.research_input_id == latest.c.research_input_id,
                ResearchAssessment.created_at == latest.c.max_created_at,
            ),
        )
        .order_by(ResearchAssessment.created_at.desc())
    ).all()
    return [_assessment_summary(assessment, research_input) for assessment, research_input in rows]


def _project_detail(session: Session, project: ResearchProject) -> ResearchProjectOut:
    input_ids = list(
        session.execute(
            select(ResearchProjectInput.research_input_id)
            .where(ResearchProjectInput.project_id == project.id)
            .order_by(ResearchProjectInput.created_at.desc())
        ).scalars()
    )
    return ResearchProjectOut(
        id=project.id,
        name=project.name,
        notes=project.notes,
        tags=project.tags,
        created_at=project.created_at,
        updated_at=project.updated_at,
        assessments=_latest_assessments(session, input_ids),
    )


@router.post("", response_model=ResearchProjectSummaryOut, status_code=status.HTTP_201_CREATED)
def create_project(payload: ResearchProjectCreate, session: Session = Depends(get_session)) -> ResearchProjectSummaryOut:
    project = ResearchProject(name=payload.name, notes=payload.notes, tags=payload.tags)
    session.add(project)
    session.commit()
    return ResearchProjectSummaryOut(
        id=project.id,
        name=project.name,
        notes=project.notes,
        tags=project.tags,
        created_at=project.created_at,
        updated_at=project.updated_at,
        assessment_count=0,
    )


@router.get("", response_model=list[ResearchProjectSummaryOut])
def list_projects(session: Session = Depends(get_session)) -> list[ResearchProjectSummaryOut]:
    rows = session.execute(
        select(ResearchProject, func.count(ResearchProjectInput.research_input_id))
        .outerjoin(ResearchProjectInput, ResearchProjectInput.project_id == ResearchProject.id)
        .group_by(ResearchProject.id)
        .order_by(ResearchProject.updated_at.desc(), ResearchProject.created_at.desc())
    ).all()
    return [
        ResearchProjectSummaryOut(
            id=project.id,
            name=project.name,
            notes=project.notes,
            tags=project.tags,
            created_at=project.created_at,
            updated_at=project.updated_at,
            assessment_count=count,
        )
        for project, count in rows
    ]


@router.get("/{project_id}", response_model=ResearchProjectOut)
def get_project(project_id: uuid.UUID, session: Session = Depends(get_session)) -> ResearchProjectOut:
    return _project_detail(session, _project_or_404(session, project_id))


@router.patch("/{project_id}", response_model=ResearchProjectSummaryOut)
def update_project(
    project_id: uuid.UUID, payload: ResearchProjectUpdate, session: Session = Depends(get_session)
) -> ResearchProjectSummaryOut:
    project = _project_or_404(session, project_id)
    changes = payload.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(project, field, value)
    project.updated_at = datetime.now(UTC)
    assessment_count = session.execute(
        select(func.count(ResearchProjectInput.research_input_id)).where(ResearchProjectInput.project_id == project.id)
    ).scalar_one()
    session.commit()
    return ResearchProjectSummaryOut(
        id=project.id,
        name=project.name,
        notes=project.notes,
        tags=project.tags,
        created_at=project.created_at,
        updated_at=project.updated_at,
        assessment_count=assessment_count,
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    project = _project_or_404(session, project_id)
    session.delete(project)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{project_id}/assessments/{research_input_id}", status_code=status.HTTP_204_NO_CONTENT)
def add_assessment_to_project(
    project_id: uuid.UUID, research_input_id: uuid.UUID, session: Session = Depends(get_session)
) -> Response:
    project = _project_or_404(session, project_id)
    if session.get(ResearchInput, research_input_id) is None:
        raise HTTPException(status_code=404, detail=f"No research input with id {research_input_id}")
    existing = session.get(ResearchProjectInput, (project.id, research_input_id))
    if existing is None:
        session.add(ResearchProjectInput(project_id=project.id, research_input_id=research_input_id))
        project.updated_at = datetime.now(UTC)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{project_id}/assessments/{research_input_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_assessment_from_project(
    project_id: uuid.UUID, research_input_id: uuid.UUID, session: Session = Depends(get_session)
) -> Response:
    _project_or_404(session, project_id)
    membership = session.get(ResearchProjectInput, (project_id, research_input_id))
    if membership is None:
        raise HTTPException(status_code=404, detail="Assessment is not in this project")
    session.delete(membership)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
