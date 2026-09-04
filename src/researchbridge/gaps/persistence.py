"""Persists CandidateGapDrafts as CandidateGap + CandidateGapEvidence rows.

Always status="pending" - the human-in-the-loop rule for gap detection
isn't optional here. Nothing this module writes should be presented to a
user as a validated finding. gap_status/resolution_note and each evidence
row's classification are written as computed by gaps/detect.py - this
module makes no judgment calls of its own, it's pure persistence.

Also mirrors each saved gap into the analysis_claims structured-reasoning
layer (Sec 16) via gaps/claims.py - additive, see that module's docstring.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, CandidateGapEvidence
from researchbridge.gaps.claims import save_claim_for_gap
from researchbridge.gaps.detect import DETECTION_METHOD, CandidateGapDraft


def save_candidate_gaps(
    session: Session, drafts: list[CandidateGapDraft], similarity_threshold: float
) -> list[CandidateGap]:
    saved: list[CandidateGap] = []
    for draft in drafts:
        gap = CandidateGap(
            seed_paper_id=draft.seed_paper_id,
            observation=draft.observation,
            gap_type="inference",
            status="pending",
            contributing_paper_count=draft.contributing_paper_count,
            similarity_threshold=similarity_threshold,
            detection_method=DETECTION_METHOD,
            gap_status=draft.gap_status,
            resolution_note=draft.resolution_note,
        )
        session.add(gap)
        session.flush()  # populate gap.id for the evidence links below

        for evidence_id in draft.evidence_ids:
            classification = draft.evidence_roles[evidence_id]
            session.add(
                CandidateGapEvidence(
                    candidate_gap_id=gap.id,
                    evidence_id=evidence_id,
                    claim_role=classification.role if classification.role != "motivation" else None,
                    self_resolution_signal=classification.self_resolution,
                    field_scope_signal=classification.field_scope,
                    own_contribution_overlap=classification.own_contribution_overlap,
                )
            )
        save_claim_for_gap(session, gap, draft)
        saved.append(gap)

    session.commit()
    return saved
