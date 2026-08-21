"""Persists CandidateGapDrafts as CandidateGap + CandidateGapEvidence rows.

Always status="pending" - Sec 35/44's human-in-the-loop rule for gap
detection isn't optional here. Nothing this module writes should be
presented to a user as a validated finding.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, CandidateGapEvidence
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
        )
        session.add(gap)
        session.flush()  # populate gap.id for the evidence links below

        for evidence_id in draft.evidence_ids:
            session.add(CandidateGapEvidence(candidate_gap_id=gap.id, evidence_id=evidence_id))
        saved.append(gap)

    session.commit()
    return saved
