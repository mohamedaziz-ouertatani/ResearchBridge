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
from researchbridge.api.schemas import AnalysisClaimPage, ExtractedClaimPage
from researchbridge.api.serializers import to_claim_details, to_extracted_claim_list_items
from researchbridge.db.models import AnalysisClaim, Evidence, ExtractedClaim, Paper

router = APIRouter(prefix="/api/claims")

MAX_LIMIT = 100
VALID_STATUSES = {"pending", "approved", "rejected"}
VALID_CLAIM_TYPES = {"fact", "inference", "hypothesis", "opportunity", "speculation"}
VALID_SOURCE_TABLES = {"candidate_gaps", "research_assessments"}

# Sec 28's raw per-paper extraction fields - a completely different
# vocabulary from VALID_CLAIM_TYPES above (which is the Sec 16 analysis-
# claims layer). Kept in sync with extraction/heuristic.py's _CUE_PHRASES
# keys plus "problem" (heuristic.py's own abstract-opening-sentence
# fallback) - no shared constant exists between the two modules to import
# from without creating an api -> extraction dependency for one frozenset.
VALID_EXTRACTED_CLAIM_TYPES = {
    "problem", "method", "research_question", "main_contribution",
    "limitations", "results", "dataset", "research_gap", "applications",
}


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


# A separate router (distinct prefix - APIRouter can't mix top-level paths
# under one prefix) for extracted_claims (Sec 28), a different table from
# analysis_claims above - see VALID_EXTRACTED_CLAIM_TYPES's own comment on
# why the two vocabularies don't overlap. Kept in this file rather than a
# new one since both power the same claims-tab frontend page.
extracted_claims_router = APIRouter(prefix="/api/extracted-claims")


@extracted_claims_router.get("", response_model=ExtractedClaimPage)
def list_extracted_claims(
    session: Session = Depends(get_session),
    claim_type: str | None = Query(None, description="problem, method, research_question, main_contribution, "
                                    "limitations, results, dataset, research_gap, or applications"),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> ExtractedClaimPage:
    if claim_type is not None and claim_type not in VALID_EXTRACTED_CLAIM_TYPES:
        raise HTTPException(
            status_code=422, detail=f"claim_type must be one of {sorted(VALID_EXTRACTED_CLAIM_TYPES)}"
        )

    # Excludes extraction_method="stub": synthetic placeholder rows, never
    # real content - same rule as serializers.py::to_claims().
    base_filters = [Evidence.extraction_method != "stub"]
    if claim_type is not None:
        base_filters.append(ExtractedClaim.claim_type == claim_type)

    count_query = (
        select(func.count(ExtractedClaim.id))
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(*base_filters)
    )
    total = session.execute(count_query).scalar_one()

    query = (
        select(ExtractedClaim, Evidence, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(*base_filters)
        .order_by(ExtractedClaim.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = session.execute(query).all()

    return ExtractedClaimPage(
        items=to_extracted_claim_list_items(rows), total=total, limit=limit, offset=offset
    )
