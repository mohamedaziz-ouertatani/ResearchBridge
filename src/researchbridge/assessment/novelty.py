"""Evidence-grounded novelty assessment (blueprint Sec 2A/18-19).

Deterministic and explainable by design - no LLM call. novelty_level is
derived purely from (a) how semantically close the nearest *evidenced*
retrieved paper is, reusing the cosine distance retrieval already
computes, and (b) whether any retrieved paper actually has usable
(non-stub) extracted claims to ground the reasoning in - OR, when
per-dimension evidence coverage is available (assessment/coverage.py), an
aggregate signal over every dimension of the idea instead of just the
single nearest paper. See "Never claims absolute novelty" below - both
paths are still a corpus-similarity signal, never proof of originality.

Never claims absolute novelty (Sec 18/34): "high" only ever means "no
closely matching paper was found in the retrieved literature sample" -
explicitly hedged in the reasoning text, never "this idea is genuinely
novel". The corpus is one CS/AI/ML slice, not the full literature.

Thresholds are corpus-calibrated, not guessed: ProximityGauge.tsx's own
calibration note observes real/relevant hits land roughly 0.35-0.85
cosine distance for this corpus/embedder; NEAR_DISTANCE reuses gaps/
cluster.py's already-measured 0.35 similarity threshold, and FAR_DISTANCE
is the midpoint of that same empirical band.

Checked against real queries against the live corpus (not just guessed):
a near-identical idea landed at 0.231 (low), two topically CS/AI-adjacent
but unmatched ideas ("real-time fraud detection with graph transformers",
"quantum-assisted cat chess") landed at 0.535-0.563 (medium), and three
genuinely off-topic queries (gibberish, pizza-topping history, bread
baking) landed at 0.713-0.745 (high) - a clean separation right where
these thresholds expect it. Still provisional pending enough real
ResearchAssessments to check against actual human judgment, not just
this one spot-check.

Dimension-aware aggregate rule (see docs/superpowers/plans/2026-08-29-
dimension-aware-assessment-coverage.md, constraint 4): deliberately NOT
"any not_found dimension -> novel" - that would let a single unmatched
dimension dominate the verdict. Instead it's a FRACTION over every scored
dimension (excluding not_assessed ones), so a handful of missing
dimensions in an otherwise well-covered idea doesn't flip the result, and
an idea whose dimensions are almost entirely covered - even without one
single paper matching the whole combination - reads as low novelty for a
real reason (recombination of known pieces), not a nearest-paper
coincidence. When every dimension is not_assessed (no relevant papers to
check dimensions against), this falls back to the original nearest-
distance-only rule unchanged - insufficient dimension-level evidence
doesn't mean the single-paper signal should be discarded too.

The 0.7/0.5/0.3 fraction thresholds (item 7 of the assessment hardening
list - re-checked after being flagged as "a first-pass heuristic, same
status as every other threshold in this file when first written"):
checked against all 23 real ResearchInputs with at least one scored
dimension (excluding not_assessed-only cases) in the live corpus - the
covered/n fraction spans the full range from 0.29 to 1.00 with no gap or
cluster suggesting a different cut point, and the current thresholds
produce a genuinely balanced, non-degenerate 3-way split: 11 "low", 11
"medium", 1 "high" - not "always medium" or a lopsided pile in one
bucket, which is the failure mode a badly-placed threshold would show up
as. There's no independent ground-truth "true novelty" label to compute
precision/recall against here (unlike extraction/evaluation.py's
DEFAULT_SIMILARITY_THRESHOLD, which has hand-written annotations to check
against), so this is a distributional check, not a P/R-validated
constant - if a future pass gathers real human-judged novelty labels,
recalibrate against those directly rather than trusting this spread
argument indefinitely.

Checked against the real corpus, not just guessed (plan Task 12): the
fraud/federated-learning worked example (7 scored dimensions, 4
established, 1 partially_addressed, 1 weak_evidence, 1 not_found) landed
covered=5/7=0.714, established=4/7=0.571 -> "low", matching the intuitive
read of that idea (the individual pieces - federated learning, privacy,
fraud detection - are each well-studied; the novel part is the specific
combination plus concept drift, which the coverage table surfaces
explicitly rather than hiding behind one nearest-distance number). A
4-dimension LLM-agent-planning idea (2 established, 1 weak, 1 not_found;
covered=2/4=0.5) landed in the middle band -> "medium", correctly not
tipping either direction on a genuinely mixed signal. Two genuinely
off-topic ideas (medieval-bread-baking-and-pizza-toppings; pure gibberish)
retrieved nothing within FAR_DISTANCE at all, so every dimension read
not_assessed and the rule correctly fell back to the legacy nearest-
distance path -> "high", not a fabricated aggregate verdict from zero
evidence. See coverage.py's DIMENSION_MATCH_SIMILARITY docstring for the
one real edge case found (a degenerate single-dimension idea) and why it
wasn't fixed by moving these thresholds - the failure was in dimension
extraction granularity, not this aggregation rule.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from researchbridge.assessment.coverage import DimensionCoverage

NEAR_DISTANCE = 0.35
FAR_DISTANCE = 0.65
TOP_N_FOR_EVIDENCE = 3

ClaimRecord = tuple[str, str, uuid.UUID]  # (claim_type, text, evidence_id)
EvidencedPaper = tuple[str, float, list[ClaimRecord]]  # (paper_title, distance, claims)

_COVERED_STATUSES = frozenset({"established", "partially_addressed"})


@dataclass
class NoveltyResult:
    level: str  # high | medium | low | not_assessed
    reasoning: str
    evidence_ids: list[uuid.UUID]
    dimension_coverage: list[DimensionCoverage] = field(default_factory=list)


def assess_novelty(
    papers_by_distance: list[EvidencedPaper],
    dimension_coverages: list[DimensionCoverage] | None = None,
) -> NoveltyResult:
    """papers_by_distance: every retrieved paper paired with its distance and
    non-stub claims, sorted nearest-first. Papers with no claims are
    filtered out here - they can't ground any reasoning.

    dimension_coverages (optional): per-dimension evidence coverage from
    assessment/coverage.py. When omitted, or when every dimension in it is
    "not_assessed", behavior is identical to the original nearest-distance-
    only rule - this parameter is a strict additive extension, not a
    replacement, so every existing caller keeps working unchanged."""
    evidenced = [p for p in papers_by_distance if p[2]]
    if not evidenced:
        return NoveltyResult(
            level="not_assessed",
            reasoning=(
                "No retrieved paper had usable extracted claims to compare against, "
                "so novelty cannot be assessed from evidence alone."
            ),
            evidence_ids=[],
        )

    nearest_title, nearest_distance, _nearest_claims = evidenced[0]
    supporting = evidenced[:TOP_N_FOR_EVIDENCE]
    evidence_ids = [evidence_id for _title, _distance, claims in supporting for _type, _text, evidence_id in claims]

    scored_coverages = [c for c in (dimension_coverages or []) if c.status != "not_assessed"]

    if scored_coverages:
        level, reasoning = _aggregate_from_coverage(scored_coverages)
    else:
        level, reasoning = _from_nearest_distance(nearest_title, nearest_distance)

    if dimension_coverages:
        lines = "\n".join(f"  {c.dimension} -> {c.status}" for c in dimension_coverages)
        reasoning = f"{reasoning}\n\nDimension coverage:\n{lines}"

    return NoveltyResult(
        level=level, reasoning=reasoning, evidence_ids=evidence_ids, dimension_coverage=dimension_coverages or []
    )


def _from_nearest_distance(nearest_title: str, nearest_distance: float) -> tuple[str, str]:
    if nearest_distance <= NEAR_DISTANCE:
        return "low", (
            f'The closest retrieved paper, "{nearest_title}", is a close semantic match '
            f"(distance {nearest_distance:.3f}), suggesting this idea substantially overlaps "
            f"with existing work already found in the corpus."
        )
    if nearest_distance >= FAR_DISTANCE:
        return "high", (
            f"No closely matching paper was found in the retrieved literature - the nearest, "
            f'"{nearest_title}", is comparatively distant (distance {nearest_distance:.3f}). '
            f"This suggests limited directly related prior work within this corpus, not "
            f"confirmed novelty against the wider scientific literature: the corpus covers "
            f"only a specific CS/AI/ML slice, and this reflects the retrieved sample, not an "
            f"exhaustive search."
        )
    return "medium", (
        f'The closest retrieved paper, "{nearest_title}", is moderately related '
        f"(distance {nearest_distance:.3f}) - some overlap exists with retrieved "
        f"literature, but not a close match."
    )


def _aggregate_from_coverage(scored_coverages: list[DimensionCoverage]) -> tuple[str, str]:
    n = len(scored_coverages)
    covered = sum(1 for c in scored_coverages if c.status in _COVERED_STATUSES)
    established = sum(1 for c in scored_coverages if c.status == "established")

    if covered / n >= 0.7 and established / n >= 0.5:
        return "low", (
            f"{established} of {n} dimensions of this idea are individually well-represented "
            f"(established) in the retrieved literature, suggesting substantial overlap with "
            f"existing work even if no single retrieved paper is a close overall match."
        )
    if covered / n <= 0.3:
        return "high", (
            f"Only {covered} of {n} dimensions of this idea have supporting evidence in the "
            f"retrieved literature sample. This reflects limited directly related prior work "
            f"within this corpus, not confirmed novelty against the wider scientific "
            f"literature: the corpus covers only a specific CS/AI/ML slice, and this reflects "
            f"the retrieved sample, not an exhaustive search."
        )
    return "medium", (
        f"{covered} of {n} dimensions of this idea have some supporting evidence in the "
        f"retrieved literature - partial overlap with existing work, not a close match on "
        f"every dimension."
    )
