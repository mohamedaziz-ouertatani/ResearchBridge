"""Checks whether analysis_claims confidence buckets (Sec 16) correlate with
real human review outcomes - the Sec 28 confidence-bucket check ("check
whether confidence buckets are actually meaningful... if they don't
correlate, the rule is providing false reassurance"), applied to the claims
layer rather than extraction.

This is scaffolding, not a calibration result: a meaningful correlation
check needs real variance across confidence buckets in the reviewed sample.
As of this module's creation, every gap-derived claim's confidence is a
hardcoded "medium" (see gaps/claims.py) - there is no second bucket to
compare against yet, so this will report exactly one bucket until gap
claim confidence is ever varied. Running this now still has real value: it
establishes the reporting mechanism and the reviewed-sample count, so
whoever varies gap claim confidence later has an existing tool to check the
result against, rather than building this from scratch once the question
becomes answerable. See gaps/cli_claims_calibration.py for the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import AnalysisClaim, CandidateGap

RATING_FIELDS = (
    "correctness_rating",
    "relevance_rating",
    "novelty_rating",
    "evidence_support_rating",
    "usefulness_rating",
)


@dataclass
class ConfidenceBucketStats:
    confidence: str
    sample_count: int
    """Reviewed (approved or rejected) gaps whose claim falls in this
    confidence bucket - not the count of gaps with every rating field set,
    since a reviewer can approve/reject without rating (see CandidateGap's
    docstring)."""
    mean_ratings: dict[str, float]
    """One entry per Sec 44 rating dimension that has at least one non-NULL
    value in this bucket - a dimension with zero ratings set across the
    whole bucket is omitted rather than reported as 0.0."""


def gap_claim_confidence_vs_ratings(session: Session) -> list[ConfidenceBucketStats]:
    """Groups reviewed CandidateGaps by their linked analysis_claims.confidence
    bucket and reports the mean of each Sec 44 rating dimension per bucket.

    Only gaps with status in (approved, rejected) count as "reviewed" - a
    still-pending gap has no human judgment to correlate against yet.
    """
    rows = session.execute(
        select(AnalysisClaim, CandidateGap)
        .join(CandidateGap, CandidateGap.id == AnalysisClaim.source_id)
        .where(AnalysisClaim.source_table == "candidate_gaps", CandidateGap.status.in_(["approved", "rejected"]))
    ).all()

    by_confidence: dict[str, list[CandidateGap]] = {}
    for claim, gap in rows:
        by_confidence.setdefault(claim.confidence, []).append(gap)

    stats: list[ConfidenceBucketStats] = []
    for confidence, gaps in sorted(by_confidence.items()):
        mean_ratings: dict[str, float] = {}
        for field in RATING_FIELDS:
            values = [getattr(gap, field) for gap in gaps if getattr(gap, field) is not None]
            if values:
                mean_ratings[field] = sum(values) / len(values)
        stats.append(ConfidenceBucketStats(confidence=confidence, sample_count=len(gaps), mean_ratings=mean_ratings))
    return stats
