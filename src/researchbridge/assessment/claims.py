"""Mirrors a ResearchAssessment's plain-text report fields into analysis_claims
(Sec 16) - the second, larger slice building on gaps/claims.py's precedent.

Scope is narrowed to the assessment's plain-text fields: comparison_summary,
novelty_reasoning, research_gap_text (only when input-specific, see below),
technical_feasibility_reasoning, risks_and_limitations. potential_applications
and potential_opportunities are JSONB lists of structured items rather than
one prose statement each - mapping those onto claims (one per list item) is
a further, not-yet-scoped step.

research_gap_text is mirrored ONLY when research_gap_source ==
"input_specific". When the assessment reused an existing candidate_gaps row
(research_gap_source == "reused_candidate_gap"), that gap already has its
own analysis_claims row via gaps/claims.py - writing a second one here would
duplicate the same inference under two different source_ids.

confidence on every claim written here is the assessment's own overall
`confidence` (already a high/medium/low judgment grounded in the evidence
backing the whole assessment - see assessment/recommendation.py), not a
fresh per-field number - same "don't invent a finer-grained confidence than
we can justify" reasoning as gaps/claims.py's hardcoded "medium".

status is always "pending": ResearchAssessment has no approve/reject review
state analogous to CandidateGap.status (only a `human_reviewed` boolean and
a pending/running/completed/failed `status`), so there is nothing to sync
these claims to - they stay pending indefinitely.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from researchbridge.db.models import AnalysisClaim, ClaimEvidence, ResearchAssessment

SOURCE_TABLE = "research_assessments"

# role -> claim_type. "comparison" summarizes what retrieved papers directly
# show (fact); everything else here is reasoning derived from that evidence
# (inference). None of these plain-text roles fit "hypothesis"/"opportunity"/
# "speculation" - those belong to the JSONB application/opportunity fields
# this slice doesn't cover yet.
CLAIM_TYPE_BY_ROLE = {
    "comparison": "fact",
    "novelty": "inference",
    "research_gap": "inference",
    "feasibility": "inference",
    "risk": "inference",
}


def save_claims_for_assessment(
    session: Session,
    assessment: ResearchAssessment,
    texts_by_role: dict[str, str | None],
    evidence_by_role: dict[str, list[uuid.UUID]],
) -> list[AnalysisClaim]:
    """Creates one AnalysisClaim per populated role in texts_by_role.

    Skips a role with no text (NULL field - Sec 22, never invent a claim for
    an unassessed field) or no evidence_ids (nothing to ground it in). Does
    not commit - the caller (assessment/build.py) commits once.
    """
    saved: list[AnalysisClaim] = []
    for role, claim_type in CLAIM_TYPE_BY_ROLE.items():
        text = texts_by_role.get(role)
        evidence_ids = evidence_by_role.get(role) or []
        if not text or not evidence_ids:
            continue

        claim = AnalysisClaim(
            claim_type=claim_type,
            claim_text=text,
            confidence=assessment.confidence or "medium",
            status="pending",
            source_table=SOURCE_TABLE,
            source_id=assessment.id,
        )
        session.add(claim)
        session.flush()  # populate claim.id for the evidence links below

        for evidence_id in evidence_ids:
            session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence_id, relationship="supports"))
        saved.append(claim)

    return saved
