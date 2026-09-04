"""Standalone read access to analysis_claims (Sec 16).

A claim is otherwise only reachable nested inside its parent gap
(CandidateGapOut.claim) or assessment (ResearchAssessmentOut.claims) - this
router lets a caller list/filter claims across the whole corpus without
already knowing which gap or assessment produced them. Read-only: a
claim's status is never set directly here, only ever synced from its
parent's own review action (gaps_routes.py::review_gap,
assessment_routes.py::review_assessment).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.schemas import AnalysisClaimPage
from researchbridge.api.serializers import to_claim_details
from researchbridge.db.models import AnalysisClaim

router = APIRouter(prefix="/api/claims")

MAX_LIMIT = 100
VALID_STATUSES = {"pending", "approved", "rejected"}
VALID_CLAIM_TYPES = {"fact", "inference", "hypothesis", "opportunity", "speculation"}
VALID_SOURCE_TABLES = {"candidate_gaps", "research_assessments"}


@router.get("", response_model=AnalysisClaimPage)
def list_claims(
    session: Session = Depends(get_session),
    status: str | None = Query(None, description="pending, approved, or rejected"),
    claim_type: str | None = Query(None, description="fact, inference, hypothesis, opportunity, or speculation"),
    source_table: str | None = Query(None, description="candidate_gaps or research_assessments"),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> AnalysisClaimPage:
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUSES)}")
    if claim_type is not None and claim_type not in VALID_CLAIM_TYPES:
        raise HTTPException(status_code=422, detail=f"claim_type must be one of {sorted(VALID_CLAIM_TYPES)}")
    if source_table is not None and source_table not in VALID_SOURCE_TABLES:
        raise HTTPException(status_code=422, detail=f"source_table must be one of {sorted(VALID_SOURCE_TABLES)}")

    query = select(AnalysisClaim)
    count_query = select(func.count(AnalysisClaim.id))
    if status is not None:
        query = query.where(AnalysisClaim.status == status)
        count_query = count_query.where(AnalysisClaim.status == status)
    if claim_type is not None:
        query = query.where(AnalysisClaim.claim_type == claim_type)
        count_query = count_query.where(AnalysisClaim.claim_type == claim_type)
    if source_table is not None:
        query = query.where(AnalysisClaim.source_table == source_table)
        count_query = count_query.where(AnalysisClaim.source_table == source_table)

    total = session.execute(count_query).scalar_one()
    claims = list(
        session.execute(query.order_by(AnalysisClaim.created_at.desc()).limit(limit).offset(offset)).scalars()
    )

    return AnalysisClaimPage(items=to_claim_details(session, claims), total=total, limit=limit, offset=offset)
