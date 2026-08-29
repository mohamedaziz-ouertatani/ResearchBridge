"""Per-dimension evidence coverage over an assessment's own retrieved
neighborhood (see docs/superpowers/plans/2026-08-29-dimension-aware-
assessment-coverage.md).

Pure function, no DB access: operates entirely on the (paper_title,
distance, claims) tuples build_assessment() has already fetched via
_claims_for_paper(), the same shape novelty.py's EvidencedPaper already
uses. This is what guarantees the hard evidence-grounding invariant -
every evidence_id this module returns is one that was already attached to
a real ExtractedClaim/Evidence row before this function ever ran; nothing
here can invent one.

Status is a corroboration-count + claim-type judgment, same philosophy as
feasibility.py preferring multi-source agreement over a single data
point:
- not_assessed: no relevant paper was retrieved at all for this
  assessment - there was nothing to check the dimension against.
- not_found: relevant papers exist, but none has a claim that matches
  this dimension - evidence was sufficient to investigate, nothing
  supported it.
- weak_evidence: exactly one relevant paper has a matching claim -
  single-source, uncorroborated.
- partially_addressed: two or more relevant papers have a matching claim,
  but only of "concern" types (limitations/problem/research_gap) - the
  dimension is acknowledged as an open issue, not shown as solved.
- established: two or more relevant papers have a matching claim, and at
  least one is an "affirmative" type (method/dataset/main_contribution/
  results/applications) - the dimension has documented coverage, not just
  documented concern.

Never claims the input idea itself is established/addressed - only that
the retrieved literature sample does or doesn't discuss this dimension.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from researchbridge.assessment.dimensions import IdeaDimension
from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.embedding.base import Embedder

ClaimRecord = tuple[str, str, uuid.UUID]  # (claim_type, text, evidence_id)
EvidencedPaper = tuple[str, float, list[ClaimRecord]]  # (paper_title, distance, claims)

# First-pass value, same status as every other threshold in this package -
# see the plan's Task 12 for the live-corpus calibration pass. Starting
# point matches the ballpark semantic.py found workable for sentence-vs-
# anchor matching (MEDIUM_CONFIDENCE_SIMILARITY = 0.28).
DIMENSION_MATCH_SIMILARITY = 0.30

_AFFIRMATIVE_CLAIM_TYPES = frozenset({"method", "dataset", "main_contribution", "results", "applications"})
_CONCERN_CLAIM_TYPES = frozenset({"limitations", "problem", "research_gap"})


@dataclass
class DimensionCoverage:
    dimension: str
    status: str  # established | partially_addressed | weak_evidence | not_found | not_assessed
    evidence_ids: list[uuid.UUID] = field(default_factory=list)
    supporting_paper_titles: list[str] = field(default_factory=list)


def compute_dimension_coverage(
    dimensions: list[IdeaDimension],
    papers_by_distance: list[EvidencedPaper],
    embedder: Embedder,
    relevance_distance: float = RELEVANCE_DISTANCE,
) -> list[DimensionCoverage]:
    if not dimensions:
        return []

    relevant = [p for p in papers_by_distance if p[1] <= relevance_distance and p[2]]
    if not relevant:
        return [DimensionCoverage(dimension=d.label, status="not_assessed") for d in dimensions]

    all_claims: list[tuple[str, ClaimRecord]] = [
        (title, claim) for title, _distance, claims in relevant for claim in claims
    ]
    claim_texts = [claim[1] for _title, claim in all_claims]
    dimension_labels = [d.label for d in dimensions]
    vectors = embedder.embed_texts(dimension_labels + claim_texts)
    dimension_vectors = vectors[: len(dimension_labels)]
    claim_vectors = vectors[len(dimension_labels) :]

    results: list[DimensionCoverage] = []
    for dimension, dvec in zip(dimensions, dimension_vectors, strict=True):
        matches: list[tuple[str, ClaimRecord]] = []
        for (title, claim), cvec in zip(all_claims, claim_vectors, strict=True):
            similarity = sum(a * b for a, b in zip(dvec, cvec, strict=True))
            if similarity >= DIMENSION_MATCH_SIMILARITY:
                matches.append((title, claim))

        results.append(_status_for(dimension.label, matches))
    return results


def _status_for(label: str, matches: list[tuple[str, ClaimRecord]]) -> DimensionCoverage:
    if not matches:
        return DimensionCoverage(dimension=label, status="not_found")

    papers_by_title: dict[str, list[ClaimRecord]] = {}
    for title, claim in matches:
        papers_by_title.setdefault(title, []).append(claim)

    distinct_paper_count = len(papers_by_title)
    evidence_ids = [claim[2] for _title, claim in matches]
    titles = list(papers_by_title)

    if distinct_paper_count == 1:
        return DimensionCoverage(
            dimension=label, status="weak_evidence", evidence_ids=evidence_ids, supporting_paper_titles=titles
        )

    has_affirmative = any(
        claim[0] in _AFFIRMATIVE_CLAIM_TYPES for claims in papers_by_title.values() for claim in claims
    )
    status = "established" if has_affirmative else "partially_addressed"
    return DimensionCoverage(
        dimension=label, status=status, evidence_ids=evidence_ids, supporting_paper_titles=titles
    )
