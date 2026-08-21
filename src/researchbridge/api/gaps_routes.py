"""Candidate gap review endpoints (Phase 3, Sec 35/44).

Detection stays a deliberate CLI step (rb-gaps-detect --save) - these
routes only list and review rows that already exist, never trigger
detection. Approving/rejecting is the only way a candidate gap's status
ever leaves "pending"; nothing here scores or auto-approves anything.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.schemas import CandidateGapOut, CandidateGapPage, CandidateGapReview
from researchbridge.api.serializers import to_gaps
from researchbridge.db.models import CandidateGap

router = APIRouter(prefix="/api/gaps")

MAX_LIMIT = 100
VALID_STATUSES = {"pending", "approved", "rejected"}


@router.get("", response_model=CandidateGapPage)
def list_gaps(
    session: Session = Depends(get_session),
    status: str = Query("pending", description="pending, approved, rejected, or all"),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> CandidateGapPage:
    if status != "all" and status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUSES)} or 'all'")

    query = select(CandidateGap)
    count_query = select(func.count(CandidateGap.id))
    if status != "all":
        query = query.where(CandidateGap.status == status)
        count_query = count_query.where(CandidateGap.status == status)

    total = session.execute(count_query).scalar_one()
    gaps = list(
        session.execute(
            query.order_by(CandidateGap.created_at.desc()).limit(limit).offset(offset)
        ).scalars()
    )

    return CandidateGapPage(items=to_gaps(session, gaps), total=total, limit=limit, offset=offset)


@router.put("/{gap_id}", response_model=CandidateGapOut)
def review_gap(
    gap_id: uuid.UUID, payload: CandidateGapReview, session: Session = Depends(get_session)
) -> CandidateGapOut:
    if payload.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUSES)}")

    gap = session.get(CandidateGap, gap_id)
    if gap is None:
        raise HTTPException(status_code=404, detail=f"No candidate gap with id {gap_id}")

    gap.status = payload.status
    gap.review_note = payload.review_note
    gap.reviewed_at = None if payload.status == "pending" else datetime.now(timezone.utc)
    session.commit()

    return to_gaps(session, [gap])[0]
