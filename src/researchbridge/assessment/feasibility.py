"""Evidence-grounded technical feasibility assessment (blueprint Sec 2A/20).

Deterministic and explainable, no LLM. Sec 20's feasibility dimensions
(deployment feasibility, compute requirements, data availability,
reproducibility, ...) don't map to a dedicated extraction field the way
research_gap/applications did, so this grounds the level in a narrower,
honest proxy instead: how much documented methodology/data a genuinely
CLOSE neighborhood provides, via the method/dataset claims those papers
already have (Sec 28/29). This never claims the idea WILL work or IS
feasible in an engineering sense - only that documented technical
grounding exists (or doesn't) in the retrieved literature. A real
feasibility judgment needs more than literature text; this is a
literature-grounded signal toward one, not a substitute for it.

Distance-gated tighter than novelty/gap/applications' RELEVANCE_DISTANCE
(0.65): checked live against the real corpus, that looser gate made
"high" fire for every idea tried, including a deliberately absurd one -
almost any paper has SOME method/dataset claim, so "2+ relevant papers
have one" was really just measuring "2+ topically-adjacent papers exist,"
not real feasibility grounding. Reusing novelty.py's tighter
NEAR_DISTANCE (0.35, its own "genuinely close match" threshold) instead
requires papers to be actually close, not merely topically adjacent,
before they can corroborate feasibility.

Level is corroboration-count based, same reasoning as the rest of this
package preferring multi-source agreement over a single data point:
- not_assessed: no genuinely close paper has a real (non-stub) method or
  dataset claim to ground a judgment in.
- medium: exactly one such paper does - single-source grounding.
- high: two or more DISTINCT such papers do - multiple independent
  sources document applicable methodology/data.

Recalibration investigated and deliberately rejected (2026-09-04), after
a real assessment (a healthcare federated-learning idea) surfaced two
clearly on-topic papers - zkFL-Health, Lightweight Privacy-First
Federated Learning for Medical AI - each with a specific, first-person
method claim, sitting just past NEAR_DISTANCE (0.3843, 0.3579) and so
not counted. A corpus-wide check (38 real ResearchInputs) confirmed this
is not an anecdote: 118 near-threshold (0.35-0.45) papers across 33/38
inputs (87%) carry a genuine method/dataset claim that NEAR_DISTANCE
excludes.

That recall gap is real, but every fix tried made things worse, not
better:

1. Flat threshold loosening (0.38/0.40/0.42/0.45/0.50, tested against
   the same 38 inputs): "high" share climbs from 42% at 0.35 to 71% at
   0.38, 84% at 0.40, 97% at 0.50 - not_assessed disappears entirely by
   0.50. None of the six tested values preserve anything like 0.35's
   balanced split (not_assessed 34% / medium 24% / high 42%), the same
   "genuinely balanced, non-degenerate" property novelty.py's own
   calibration note treats as the signal of a well-placed threshold.
   Sampled false positives past ~0.42 are same-broad-domain-wrong-
   technique mismatches (e.g. a fridge recipe-suggestion ML paper
   matched to a sourdough-starter-timing idea on "IoT + ML" alone) -
   the exact failure this file's own NEAR_DISTANCE choice was
   originally meant to avoid, just re-emerging one notch later.

2. Decoupling "which papers count as evidence" from "how the count
   maps to a level" (keeping NEAR_DISTANCE as a trusted inner tier,
   admitting a wider outer tier only when a claim passes an extra
   quality bar) is the right-shaped fix in principle, but neither
   deterministic, non-LLM qualifier tried actually discriminates:
   - Reusing extraction/heuristic.py's method/dataset cue phrases ("we
     propose", "we present", ...) as the bar is nearly a no-op: with
     the outer tier extended to 0.65 every input read "high" (38/38);
     even capped at 0.40 it barely differed from a flat 0.40 cutoff
     with no filter at all (31 vs 32 "high" out of 38). Root cause:
     that cue list is close to what HeuristicExtractor itself used to
     tag the sentence "method" in the first place, so it doesn't add
     information beyond claim_type.
   - Embedding the claim's own sentence against the query (instead of
     the paper's whole title+abstract) was inconsistent in both
     directions: it correctly flagged one generic survey-style claim as
     far off-topic, but also pushed two genuinely good method claims
     (Lightweight Privacy-First, RLPF) farther away than their already-
     marginal paper-level distance, while pulling one bad match (a
     stress-detection wearable paper, coincidentally sharing
     "detection"/"wearable" vocabulary) closer.

Conclusion: NEAR_DISTANCE stays 0.35. The corroboration-count levels
only mean something if "high" stays a real minority outcome, not the
default; every tested alternative bought recall by breaking that. If a
future signal (e.g. checking a paper's own full-text section, once the
fulltext pipeline - Sec 46 - covers more of the corpus, rather than only
its title+abstract) can be shown to discriminate zkFL-Health from a
recipe-suggestion fridge paper without this tradeoff, recalibrate then -
not by picking a new constant against this one healthcare example.

Widened band added (2026-09-05), after a second investigation found a
deterministic qualifier that DOES discriminate - narrower in shape than
either fix rejected above: a paper in (NEAR_DISTANCE, WIDENED_DISTANCE]
(0.35-0.40) counts toward corroboration only if at least one of its
method/dataset claims has IDF-weighted lexical overlap with the research
input's own text >= CLAIM_OVERLAP_THRESHOLD (see claim_relevance.py).
Checked against the real corpus and every real ResearchInput: this
combination recovered 8 genuinely on-technique papers across 5 inputs
(STG-GNN/FinFraudBench for a fraud-detection-graph-transformer idea,
WeSCE/VICBench for an LLM-plus-static-analysis vulnerability idea,
Coevolutionary-FL-DP for the sepsis federated-learning idea) with ZERO
confirmed false positives, and left the overall not_assessed/medium/high
split close to baseline (33/23/44 vs 35/23/42) - nothing like the flat
widening collapse above. One confirmed near-miss was correctly rejected:
a smart-home load-monitoring paper (no federated-learning content at all)
scored 0.169, below the 0.20 bar, for a "federated learning for smart
home energy prediction" idea - a same-broad-domain-wrong-technique match,
exactly the failure this bar exists to catch.

This does NOT recover zkFL-Health / Lightweight Privacy-First - both
score well under even the loosest overlap threshold tested (0.059, 0.068
vs the 0.20 bar) against the healthcare idea they were meant to ground,
because their claims use different specific vocabulary (zero-knowledge
proofs, Trusted Execution Environments, colon histopathology) than the
query's own phrasing. That is a deep-paraphrase gap: the papers are
semantically close (which is exactly why the embedder places them at
0.36-0.38) but lexically almost disjoint from the query, and no lexical
mechanism - this one included - can bridge that. This is accepted as a
known, permanent limitation of this specific fix, not a bug: the widened
band targets mild-paraphrase/shared-vocabulary near-misses, not deep
paraphrase. A future fix for THAT gap would need a semantic (not lexical)
qualifier, which is out of scope here - see the two approaches already
rejected above for why an embedding-based claim-level qualifier was tried
and found inconsistent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.claim_relevance import CLAIM_OVERLAP_THRESHOLD, WIDENED_DISTANCE, claim_overlap
from researchbridge.assessment.novelty import NEAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

PaperWithDistance = tuple[uuid.UUID, float]
FEASIBILITY_CLAIM_TYPES = ("method", "dataset")


@dataclass
class FeasibilityResult:
    level: str  # high | medium | not_assessed
    reasoning: str
    evidence_ids: list[uuid.UUID]


def assess_technical_feasibility(
    session: Session,
    papers_by_distance: list[PaperWithDistance],
    query_text: str,
    idf: dict[str, float],
) -> FeasibilityResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns).
    query_text/idf: the research input's own text and the corpus-wide IDF
    table (see claim_relevance.build_corpus_idf), used only to admit
    widened-band (0.35 < distance <= WIDENED_DISTANCE) papers whose
    claims share enough specific vocabulary with query_text - see this
    module's docstring for the calibration behind that bar. Papers within
    NEAR_DISTANCE (0.35) are entirely unaffected by query_text/idf, exactly
    as before this widened band existed."""
    close_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    widened_ids = [
        paper_id for paper_id, distance in papers_by_distance if RELEVANCE_DISTANCE < distance <= WIDENED_DISTANCE
    ]
    candidate_ids = close_ids + widened_ids
    if not candidate_ids:
        return FeasibilityResult(
            level="not_assessed",
            reasoning=(
                "No retrieved paper was close enough in meaning to ground a feasibility "
                "judgment in, so technical feasibility cannot be assessed from the literature."
            ),
            evidence_ids=[],
        )

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.evidence_id, Evidence.text, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(candidate_ids),
            ExtractedClaim.claim_type.in_(FEASIBILITY_CLAIM_TYPES),
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return FeasibilityResult(
            level="not_assessed",
            reasoning=(
                "No relevant retrieved paper had a documented method or dataset to ground "
                "a feasibility judgment in, so technical feasibility cannot be assessed from "
                "the literature alone."
            ),
            evidence_ids=[],
        )

    claims_by_paper: dict[uuid.UUID, list[tuple[uuid.UUID, str]]] = {}
    titles_by_paper: dict[uuid.UUID, str] = {}
    for paper_id, evidence_id, text, title in rows:
        claims_by_paper.setdefault(paper_id, []).append((evidence_id, text))
        titles_by_paper[paper_id] = title

    evidence_ids: list[uuid.UUID] = []
    close_titles: list[str] = []
    widened_titles: list[str] = []

    for paper_id in close_ids:
        claims = claims_by_paper.get(paper_id)
        if not claims:
            continue
        close_titles.append(titles_by_paper[paper_id])
        evidence_ids.extend(evidence_id for evidence_id, _text in claims)

    for paper_id in widened_ids:
        claims = claims_by_paper.get(paper_id)
        if not claims:
            continue
        passing_claims = [
            (evidence_id, text) for evidence_id, text in claims if claim_overlap(query_text, text, idf) >= CLAIM_OVERLAP_THRESHOLD
        ]
        if passing_claims:
            widened_titles.append(titles_by_paper[paper_id])
            evidence_ids.extend(evidence_id for evidence_id, _text in passing_claims)

    distinct_count = len(close_titles) + len(widened_titles)
    if distinct_count == 0:
        return FeasibilityResult(
            level="not_assessed",
            reasoning=(
                "No relevant retrieved paper had a documented method or dataset to ground "
                "a feasibility judgment in, so technical feasibility cannot be assessed from "
                "the literature alone."
            ),
            evidence_ids=[],
        )

    if distinct_count == 1:
        level = "medium"
        if close_titles:
            (title,) = close_titles
            reasoning = (
                f'One relevant retrieved paper, "{title}", documents a method or dataset '
                f"applicable to this idea - some technical grounding exists in the literature, "
                f"but from a single source only."
            )
        else:
            (title,) = widened_titles
            reasoning = (
                f'One relevant retrieved paper, "{title}", grounds this via a wider match '
                f"confirmed by shared technical vocabulary with this idea - some technical "
                f"grounding exists in the literature, but from a single, less directly-matched "
                f"source only."
            )
    else:
        level = "high"
        reasoning = (
            f"{distinct_count} relevant retrieved papers document methods or datasets "
            f"applicable to this idea, suggesting demonstrated technical grounding from "
            f"multiple independent sources in the literature. This reflects documented "
            f"prior work, not a guarantee of success for this specific idea."
        )
        if widened_titles:
            reasoning += (
                f" Of these, {', '.join(widened_titles)} is a wider match, corroborated by "
                f"shared technical vocabulary with this idea, rather than a directly close one."
            )

    return FeasibilityResult(level=level, reasoning=reasoning, evidence_ids=evidence_ids)
