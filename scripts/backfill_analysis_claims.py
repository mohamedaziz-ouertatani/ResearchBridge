"""One-off backfill: creates the analysis_claims (Sec 16) row a CandidateGap
or ResearchAssessment would have gotten if it had been saved/reviewed after
gaps/claims.py and assessment/claims.py existed.

Both modules already write a claim at creation/review time going forward -
this script only covers the gap between "analysis_claims shipped" and
"every existing row that predates it". It never re-runs detection or
re-builds an assessment; it only reads what's already persisted
(CandidateGap.observation + its CandidateGapEvidence links, or a
ResearchAssessment's already-populated text fields + its
ResearchAssessmentEvidence links) and mirrors that into a claim, exactly
the same shape gaps/claims.py::save_claim_for_gap and
assessment/claims.py::save_claims_for_assessment already produce.

A gap/assessment with nothing groundable (no contributing evidence links,
or none of the 7 gradeable fields populated) is skipped, same as the live
path - never invents a claim with no evidence behind it.

Usage:
    uv run python scripts/backfill_analysis_claims.py            # dry run
    uv run python scripts/backfill_analysis_claims.py --apply    # writes
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.claims import (
    SOURCE_TABLE as ASSESSMENT_SOURCE_TABLE,
    render_applications_text,
    render_opportunities_text,
    save_claims_for_assessment,
)
from researchbridge.config import load_config
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    CandidateGapEvidence,
    ClaimEvidence,
    ResearchAssessment,
    ResearchAssessmentEvidence,
)
from researchbridge.db.session import make_engine
from researchbridge.gaps.claims import (
    CLAIM_CONFIDENCE as GAP_CLAIM_CONFIDENCE,
    CLAIM_TYPE as GAP_CLAIM_TYPE,
    SOURCE_TABLE as GAP_SOURCE_TABLE,
)


def _gaps_missing_a_claim(session: Session) -> list[CandidateGap]:
    already_claimed = select(AnalysisClaim.source_id).where(AnalysisClaim.source_table == GAP_SOURCE_TABLE)
    return list(session.execute(select(CandidateGap).where(CandidateGap.id.notin_(already_claimed))).scalars())


def _assessments_missing_a_claim(session: Session) -> list[ResearchAssessment]:
    already_claimed = select(AnalysisClaim.source_id).where(AnalysisClaim.source_table == ASSESSMENT_SOURCE_TABLE)
    return list(
        session.execute(select(ResearchAssessment).where(ResearchAssessment.id.notin_(already_claimed))).scalars()
    )


def _backfill_gap(session: Session, gap: CandidateGap) -> AnalysisClaim | None:
    evidence_ids = [
        link.evidence_id
        for link in session.execute(
            select(CandidateGapEvidence).where(CandidateGapEvidence.candidate_gap_id == gap.id)
        ).scalars()
    ]
    if not evidence_ids:
        return None

    claim = AnalysisClaim(
        claim_type=GAP_CLAIM_TYPE,
        claim_text=gap.observation,
        confidence=GAP_CLAIM_CONFIDENCE,
        status=gap.status,
        source_table=GAP_SOURCE_TABLE,
        source_id=gap.id,
    )
    session.add(claim)
    session.flush()
    for evidence_id in evidence_ids:
        session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence_id, relationship="supports"))
    return claim


def _backfill_assessment(session: Session, assessment: ResearchAssessment) -> list[AnalysisClaim]:
    evidence_by_role: dict[str, list] = {}
    for link in session.execute(
        select(ResearchAssessmentEvidence).where(ResearchAssessmentEvidence.research_assessment_id == assessment.id)
    ).scalars():
        evidence_by_role.setdefault(link.role, []).append(link.evidence_id)

    texts_by_role = {
        "comparison": assessment.comparison_summary,
        "novelty": assessment.novelty_reasoning,
        "research_gap": (
            assessment.research_gap_text if assessment.research_gap_source == "input_specific" else None
        ),
        "application": (
            render_applications_text(assessment.potential_applications) if assessment.potential_applications else None
        ),
        "opportunity": (
            render_opportunities_text(assessment.potential_opportunities)
            if assessment.potential_opportunities
            else None
        ),
        "feasibility": assessment.technical_feasibility_reasoning,
        "risk": assessment.risks_and_limitations,
    }
    return save_claims_for_assessment(session, assessment, texts_by_role, evidence_by_role)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        gaps = _gaps_missing_a_claim(session)
        assessments = _assessments_missing_a_claim(session)

        gap_results: list[tuple[CandidateGap, AnalysisClaim | None]] = [
            (gap, _backfill_gap(session, gap)) for gap in gaps
        ]
        assessment_results: list[tuple[ResearchAssessment, list[AnalysisClaim]]] = [
            (assessment, _backfill_assessment(session, assessment)) for assessment in assessments
        ]

        gap_claims_created = sum(1 for _gap, claim in gap_results if claim is not None)
        gap_skipped = sum(1 for _gap, claim in gap_results if claim is None)
        assessment_claims_created = sum(len(claims) for _assessment, claims in assessment_results)
        assessments_with_no_claims = sum(1 for _assessment, claims in assessment_results if not claims)

        print(f"checked {len(gaps)} candidate_gaps without a claim: {gap_claims_created} would get one, {gap_skipped} skipped (no evidence)")
        print(
            f"checked {len(assessments)} research_assessments without any claim: "
            f"{assessment_claims_created} claim(s) would be created across them, "
            f"{assessments_with_no_claims} have nothing groundable"
        )

        if gap_claims_created == 0 and assessment_claims_created == 0:
            session.rollback()
            return

        backup_path = Path(f"scripts/backfills/analysis_claims_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                {
                    "gap_claims": [
                        {"gap_id": str(gap.id), "claim_id": str(claim.id)}
                        for gap, claim in gap_results
                        if claim is not None
                    ],
                    "assessment_claims": [
                        {"assessment_id": str(assessment.id), "claim_id": str(claim.id)}
                        for assessment, claims in assessment_results
                        for claim in claims
                    ],
                },
                indent=2,
            )
        )
        print(f"backed up {gap_claims_created + assessment_claims_created} new claim ids to {backup_path}")

        if not args.apply:
            session.rollback()
            print("dry run only - pass --apply to write changes")
            return

        session.commit()
        print(f"created {gap_claims_created} gap claim(s) and {assessment_claims_created} assessment claim(s)")


if __name__ == "__main__":
    main()
