"""Mirrors a ResearchAssessment's report fields into analysis_claims (Sec 16).

Covers all seven gradeable fields: comparison_summary, novelty_reasoning,
research_gap_text (only when input-specific, see below), the JSONB
potential_applications/potential_opportunities lists, technical_feasibility_
reasoning, and risks_and_limitations. The five plain-text fields each become
one claim with claim_text set to the field's own text verbatim (so the web
report and export can match a claim back to its field by exact text - see
AssessmentReport.tsx's claimForText / export.py's _claim_for_text). The two
JSONB list fields become one claim each too, with claim_text a deterministic
join of the list (see _render_applications/_render_opportunities) - nothing
elsewhere currently tries to match those two by exact text, so the join
format only has to be internally consistent, not mirrored in the frontend.

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

status mirrors ResearchAssessment.human_reviewed (see sync_claim_status,
called from api/assessment_routes.py::review_assessment): "approved" once a
human has looked at the assessment, "pending" otherwise. Not a real
per-claim review (a human reviews the whole assessment, not each field
individually), but it's the closest honest analog to CandidateGap.status's
approve/reject sync - a claim from a never-reviewed assessment should not
read the same as one a human has actually seen.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from researchbridge.db.models import AnalysisClaim, ClaimEvidence, ResearchAssessment

SOURCE_TABLE = "research_assessments"

# role -> claim_type. "comparison" summarizes what retrieved papers directly
# show (fact). "application"/"opportunity" are plausible practical uses of a
# capability - Sec 16's "opportunity" bucket, not a certainty. Everything
# else is reasoning derived from evidence (inference).
CLAIM_TYPE_BY_ROLE = {
    "comparison": "fact",
    "novelty": "inference",
    "research_gap": "inference",
    "application": "opportunity",
    "opportunity": "opportunity",
    "feasibility": "inference",
    "risk": "inference",
}


def render_applications_text(applications: list[dict]) -> str:
    """Same join format as assessment/export.py's applications_body, kept
    here as the single source of truth for what an "application" claim's
    text looks like."""
    return "\n".join(f"- {app['application']} (source: {app['source_paper']})" for app in applications)


def render_opportunities_text(opportunities: list[dict]) -> str:
    return "\n".join(f"{o['tier']}: {o['opportunity']}" for o in opportunities)


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


def sync_claim_status(session: Session, assessment: ResearchAssessment) -> None:
    """Propagates assessment.human_reviewed onto every claim linked to it.

    Called from api/assessment_routes.py::review_assessment after
    human_reviewed changes, so the claims layer never silently disagrees
    with whether a human has actually looked at this assessment. An
    assessment with no linked claims (nothing was groundable, or it
    predates this module) is a no-op, not an error.
    """
    claims = session.query(AnalysisClaim).filter_by(source_table=SOURCE_TABLE, source_id=assessment.id).all()
    new_status = "approved" if assessment.human_reviewed else "pending"
    for claim in claims:
        claim.status = new_status
