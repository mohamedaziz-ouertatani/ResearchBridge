"""Translates a persisted CandidateGap into an AnalysisClaim (Sec 16).

Additive, not a replacement: CandidateGap/CandidateGapEvidence stay the
source of truth for gap review (api/gaps_routes.py, the admin panel) exactly
as before. This module only mirrors an already-saved gap into the
claim_type="inference" / claim_evidence structured-reasoning layer, so
callers who want Evidence -> Inference -> Opportunity as an explicit,
typed chain don't have to parse CandidateGap.observation prose to get it.

Confidence is fixed at "medium" for this first slice: these are inferences
synthesized across a cluster that already cleared gaps/cluster.py's minimum
size and evidence bar (stronger than an unanchored guess), but produced by a
deterministic heuristic rather than a calibrated process (stronger claim
than "low" implies). Revisit once benchmark evaluation (Sec 28) can
calibrate this per the blueprint's confidence-bucket check.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from researchbridge.db.models import AnalysisClaim, CandidateGap, ClaimEvidence
from researchbridge.gaps.detect import CandidateGapDraft

SOURCE_TABLE = "candidate_gaps"
CLAIM_TYPE = "inference"
CLAIM_CONFIDENCE = "medium"


def save_claim_for_gap(session: Session, gap: CandidateGap, draft: CandidateGapDraft) -> AnalysisClaim:
    """Creates the AnalysisClaim + ClaimEvidence rows mirroring `gap`.

    `gap` must already be flushed (has an id). Does not commit - the caller
    (gaps/persistence.py) commits once after all drafts are processed.
    """
    claim = AnalysisClaim(
        claim_type=CLAIM_TYPE,
        claim_text=gap.observation,
        confidence=CLAIM_CONFIDENCE,
        status=gap.status,
        source_table=SOURCE_TABLE,
        source_id=gap.id,
    )
    session.add(claim)
    session.flush()  # populate claim.id for the evidence links below

    for evidence_id in draft.evidence_ids:
        session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence_id, relationship="supports"))

    return claim


def sync_claim_status(session: Session, gap: CandidateGap) -> None:
    """Propagates gap.status onto its linked AnalysisClaim, if one exists.

    Called from api/gaps_routes.py::review_gap after a reviewer changes a
    CandidateGap's status, so the two layers never silently disagree. A gap
    saved before this module existed has no linked claim - a no-op, not an
    error.
    """
    claim = session.query(AnalysisClaim).filter_by(source_table=SOURCE_TABLE, source_id=gap.id).one_or_none()
    if claim is not None:
        claim.status = gap.status
