"""Deterministic, extractive claim-level and cluster-level classification of
whether a limitations/research_gap claim is evidence of an actual unresolved
gap, versus a recurring topic or a paper's own motivation for its
contribution. No LLM anywhere - regex lexicons and cosine similarity only,
same as extraction/validation.py and gaps/cluster.py.

Two false-positive patterns motivated this module, both observed on real
production output:

1. "This paper investigates whether LLMs can help bridge this gap." - a
   paper naming a gap and, in the same breath, claiming to address it. The
   sentence contains real gap-adjacent vocabulary and is topically similar
   to other papers' gap statements, so upstream clustering (gaps/cluster.py)
   has no way to see that the author is describing their OWN contribution,
   not a still-open field-wide state.

2. Several benchmark papers each honestly motivating their own contribution
   with near-identical "existing benchmarks lack X" phrasing, where X is
   exactly what that paper's own contribution then measures. No single
   sentence uses first-person resolution language ("we propose to address
   this"), so a purely lexical self-resolution check misses it - but each
   paper's own main_contribution/results claim is a near-paraphrase of its
   own limitation claim, which a cosine-similarity check catches directly.

Neither signal is proof a gap is closed - a paper claiming "we propose X to
address Y" doesn't mean Y is solved, and a paper whose contribution is a
benchmark/evaluation of Y doesn't mean Y is measured AND solved. Each is
independently only worth one point of downgrade; only when BOTH fire
together (explicit self-resolution language AND the claim closely matches
the paper's own contribution) is a claim excluded outright as pure
motivation - see classify_claim's scoring below.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Literal

ClaimRole = Literal["anchor", "supporting", "motivation"]

GapStatus = Literal["strong_gap", "potential_gap", "known_limitation"]

# Same-sentence first-person resolution language: the author pairing a
# gap/limitation statement with their own paper's attempt to close it, in
# the same claim. Deliberately narrower than extraction/validation.py's
# _METHOD_LANGUAGE_RE (which is fine to fire on any method-flavored
# sentence) - this only fires on language that couples the gap TO this
# paper's own resolution attempt, not on method language in general.
_SELF_RESOLUTION_RE = re.compile(
    r"\bthis (paper|work|study) (proposes|introduces|presents|addresses|investigates|explores)\b"
    r"|\bwe (propose|introduce|present|address|investigate|explore) (a|an|this|the|whether)\b"
    r"|\bto (address|overcome|bridge|solve|tackle) this\b"
    r"|\b(help|helps|helping) (bridge|close|fill|address) (this|the) gap\b",
    re.IGNORECASE,
)

# Collective/field-wide framing: the sentence attributes the gap to the
# field broadly (existing/current/prior work, state of the art) rather than
# describing only this paper's own before-state. Required, alongside a
# strong gap tier and no self-resolution language, for a claim to anchor a
# strong_gap on its own (see classify_claim).
_FIELD_SCOPE_RE = re.compile(
    r"\bexisting (methods|approaches|systems|models|work|benchmarks)\b"
    r"|\bcurrent (approaches|methods|systems|models)\b"
    r"|\bprior (work|approaches|methods)\b"
    r"|\bstate[- ]of[- ]the[- ]art\b"
    r"|\ball (evaluated|tested) (methods|systems|models)\b"
    r"|\bdespite (recent|significant) (progress|advances)\b"
    r"|\bno existing\b",
    re.IGNORECASE,
)

# How similar a limitation/research_gap claim needs to be to the SAME
# paper's own main_contribution/results claims before that overlap counts
# as negative evidence (this paper's own contribution appears to already
# cover what it just named as a gap). Calibrated against the real
# 40-paper Sec 25 benchmark plus benchmark/gap_calibration_groups.yaml's
# curated same-paper self-pairs - see gaps/calibration.py and
# benchmark/gap_calibration_results.json for the full sweep (Task 14,
# re-run with a widened 0.30-0.95 threshold range after the initial
# 0.30-0.70 sweep turned out not to have plateaued yet - see below).
# The precision-first decision rule scores this against two "should NOT
# overlap" categories (genuine_unresolved_gap, n=28;
# recurring_limitation_genuinely_unresolved, n=3) and two "SHOULD overlap"
# categories (motivation_addressed, n=40; recurring_limitation_topical_
# convergence, n=3). At the starting placeholder 0.5, genuine_unresolved_gap
# had a false-positive rate of 0.607 - well over half of real, still-open
# gaps would have been wrongly demoted as "already covered by the paper's
# own contribution." Its false-positive rate falls monotonically as the
# threshold rises (0.929 at 0.30 -> 0.071 at 0.70) and, on the widened
# sweep, keeps falling past 0.70 (0.036 at 0.75/0.80) before finally
# plateauing at 0.000 from 0.85 onward - so the original 0.70 pick was
# premature; it stopped at the edge of the originally-swept range, not at
# an actual plateau. recurring_limitation_genuinely_unresolved's
# false-positive rate follows the same shape, bottoming out at 0.000
# (0/3) starting at 0.85 as well (having sat at 0.333 from 0.65-0.80).
# 0.85 is therefore the smallest threshold at which BOTH "should NOT
# overlap" categories reach their sweep-best (0.000) simultaneously.
# Condition 2 of the documented rule (should-overlap false negatives not
# at their sweep-worst) is never satisfiable anywhere in this range:
# recurring_limitation_topical_convergence's false-negative rate saturates
# at its worst value (1.000) from 0.65 all the way to 0.95, so every
# candidate threshold that clears condition 1 also sits at
# recurring_limitation_topical_convergence's worst case - per the
# documented fallback for exactly this situation, the threshold is set to
# minimize genuine_unresolved_gap's false-positive rate specifically, even
# at the cost of a higher false-negative rate elsewhere. That minimum
# (0.000, 0/28) is first reached at 0.85, where
# recurring_limitation_genuinely_unresolved is also at its minimum (0.000,
# 0/3) - at the cost of motivation_addressed's false-negative rate reaching
# 0.925 and recurring_limitation_topical_convergence's reaching 1.000
# (both worse than at 0.70, but this is the documented, deliberate
# trade-off: wrongly excluding a genuine unresolved gap is worse than
# missing a motivation-language claim, which only ever demotes a claim's
# role, never hides the cluster). 0.70 was reconsidered and rejected
# because it was not yet at genuine_unresolved_gap's sweep floor (0.071 vs.
# 0.85's 0.000) - it looked plateaued only because the originally-run
# sweep stopped there, not because the false-positive rate had actually
# leveled off. See benchmark/gap_calibration_results.json for the full
# per-threshold table (now covering 0.30-0.95).
#
# External-validity caveat for a future recalibration: this sweep measures
# similarity between hand-written annotation FIELDS (main_contribution vs.
# research_gap.remaining - full paragraphs written independently by an
# annotator), but at runtime _own_contribution_overlaps (detect.py) compares
# extracted CLAIM SPANS - single sentences lifted verbatim from the same
# paper's abstract. Two sentences from the same abstract plausibly share
# more surface vocabulary than two independently-written summary fields, so
# the real runtime similarity distribution may sit higher than what 0.85 was
# calibrated against. This is not believed to be a bug in the current
# threshold - just a reminder that whoever recalibrates this constant later
# should be aware they may be comparing against a different text
# distribution than what actually runs in production.
OWN_CONTRIBUTION_OVERLAP_THRESHOLD = 0.85

# Same idea, cross-paper: how similar a cluster's representative gap text
# needs to be to some OTHER paper's main_contribution/results claim before
# that counts as an "addressing signal" worth a reviewer note and a one-
# level downgrade (never an exclusion - see apply_addressing_downgrade).
# Calibrated against benchmark/gap_calibration_groups.yaml's curated
# cross-paper pairs - see benchmark/gap_calibration_results.json (Task 14).
# Unlike OWN_CONTRIBUTION_OVERLAP_THRESHOLD above, this signal is advisory
# only (a note plus a one-level downgrade, never an exclusion), so it is
# deliberately calibrated to minimize false warnings, NOT to maximize
# warning coverage - do not "fix" this later by lowering it to catch more
# high_value_addressing_match cases; that trades reviewer trust in the
# signal for a marginal, non-gating detection gain, which the calibration
# explicitly rejects (see Task 14's asymmetric decision rule). Across the
# entire swept range (0.30-0.95), false_warning_topical_only's
# false-positive rate was 0.0 at every single value tested (its two
# curated pairs - BioBERT-replication/Mandarin-TTS-probing and
# GPU-TPU-batch-size/hyper-ball-clustering-index - share only a coarse
# domain label, never real subject matter, and never spuriously cleared
# any tested threshold). Because the false-positive rate never rises above
# zero anywhere in the tested range, "pick the largest threshold at/near
# zero false positives" is a degenerate instruction here - it buys no
# additional false-positive protection (there is none to buy) while
# strictly destroying detection, so the value is instead the SMALLEST
# threshold that holds false_warning_topical_only's false-positive rate at
# its floor (0.0, true everywhere) while also minimizing the combined
# false-negative rate of high_value_addressing_match and
# benchmark_evaluation_partial_solution: both are tied at their combined
# minimum (0.0 + 0.5 = 0.5) at 0.30 and 0.35 alike, so 0.30 - the smaller
# of the tied pair - is used, rather than climbing to 0.70 where
# high_value_addressing_match's false-negative rate reaches 1.0 for no
# false-positive benefit. A missed addressing warning only means a
# reviewer doesn't get a "verify before treating this as fully open" note;
# a false one erodes trust in the signal until reviewers start ignoring
# it, which is worse long-term - but "conservative" here means picking the
# smallest value that already achieves zero false positives, not padding
# the margin past that point once the floor is reached. Caveat for a
# future recalibration: false_warning_topical_only and
# high_value_addressing_match/benchmark_evaluation_partial_solution each
# have only 2 curated cross-paper examples (n=2) in
# gap_calibration_groups.yaml as of this calibration - a threshold chosen
# from 2 examples per category is thin evidence even though the reasoning
# above is sound; adding more curated cross-paper pairs and re-running the
# sweep would meaningfully strengthen this constant.
ADDRESSING_SIMILARITY_THRESHOLD = 0.30


@dataclass(frozen=True)
class ClaimClassification:
    role: ClaimRole
    self_resolution: bool
    field_scope: bool
    own_contribution_overlap: float


def classify_claim(
    text: str, claim_type: str, gap_tier: str | None, own_contribution_overlap: float
) -> ClaimClassification:
    """Scores one limitations/research_gap claim toward anchor/supporting/
    motivation. Starting score is 2 for an unambiguous ("strong") tier, 1
    for everything else - a "limitations" claim has no tier of its own
    (validation.py never sets one for it) and always starts at 1, so it can
    corroborate a cluster but never single-handedly anchor a strong_gap;
    only an explicit, strong-tier research_gap claim can do that, and only
    with field-scope language and neither negative signal firing.

    self_resolution and own_contribution_overlap are each worth exactly one
    point of downgrade - deliberately not two, and deliberately not an
    automatic exclusion on their own (see module docstring: "we introduce a
    benchmark for X" is real overlap but not proof X is solved). Only a
    claim hit by BOTH drops all the way to "motivation".
    """
    self_resolution = bool(_SELF_RESOLUTION_RE.search(text))
    field_scope = bool(_FIELD_SCOPE_RE.search(text))

    score = 2 if gap_tier == "strong" else 1
    if self_resolution:
        score -= 1
    if own_contribution_overlap >= OWN_CONTRIBUTION_OVERLAP_THRESHOLD:
        score -= 1

    if score >= 2 and field_scope:
        role: ClaimRole = "anchor"
    elif score >= 1:
        role = "supporting"
    else:
        role = "motivation"

    return ClaimClassification(
        role=role,
        self_resolution=self_resolution,
        field_scope=field_scope,
        own_contribution_overlap=own_contribution_overlap,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """A plain dot product - assumes both `a` and `b` are already
    L2-normalized (true for SentenceTransformerEmbedder's output, which
    passes normalize_embeddings=True), NOT a full cosine formula with its
    own normalization built in. A future Embedder that doesn't normalize
    would silently invalidate every threshold calibrated against this
    function."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class ClassifiedMember:
    paper_id: uuid.UUID
    evidence_id: uuid.UUID
    text: str
    classification: ClaimClassification


def classify_cluster(members: list[ClassifiedMember], min_cluster_size: int) -> GapStatus | None:
    """Aggregates one cluster's classified members into a gap status, or
    None if nothing here clears the bar to be shown to a reviewer at all.

    "independent_papers" (papers whose claim is NOT itself overlap-flagged)
    is the cross-paper corroboration requirement: min_cluster_size distinct
    papers saying something gap-shaped is not enough on its own if every one
    of them is explainable by that same paper's own contribution - that's
    corroboration of a shared research topic, not of anything unresolved.
    """
    anchor_papers = {m.paper_id for m in members if m.classification.role == "anchor"}
    supporting_members = [m for m in members if m.classification.role in ("anchor", "supporting")]
    supporting_papers = {m.paper_id for m in supporting_members}
    independent_papers = {
        m.paper_id
        for m in supporting_members
        if m.classification.own_contribution_overlap < OWN_CONTRIBUTION_OVERLAP_THRESHOLD
    }

    if len(anchor_papers) >= 2:
        return "strong_gap"
    if len(anchor_papers) == 1 and len(supporting_papers) >= min_cluster_size and len(independent_papers) >= 2:
        return "potential_gap"
    if len(independent_papers) >= min_cluster_size:
        return "known_limitation"
    return None


@dataclass(frozen=True)
class AddressingMatch:
    paper_id: uuid.UUID
    paper_title: str
    text: str
    similarity: float


def find_addressing_papers(
    representative_vector: list[float],
    candidates: list[tuple[uuid.UUID, str, str, list[float]]],
    threshold: float = ADDRESSING_SIMILARITY_THRESHOLD,
) -> list[AddressingMatch]:
    """candidates is (paper_id, paper_title, claim_text, vector) for every
    main_contribution/results claim in the seed paper's neighborhood - not
    just the cluster's own contributing papers, since a DIFFERENT paper in
    the corpus already addressing the pattern is exactly the case this
    exists to surface."""
    matches = [
        AddressingMatch(paper_id=paper_id, paper_title=paper_title, text=text, similarity=similarity)
        for paper_id, paper_title, text, vector in candidates
        if (similarity := cosine_similarity(representative_vector, vector)) >= threshold
    ]
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


_DOWNGRADE: dict[GapStatus, GapStatus] = {
    "strong_gap": "potential_gap",
    "potential_gap": "known_limitation",
    "known_limitation": "known_limitation",
}


def apply_addressing_downgrade(
    status: GapStatus, matches: list[AddressingMatch]
) -> tuple[GapStatus, str | None]:
    """Negative evidence, never invalidation: a paper's contribution being
    similar to a named gap can mean it addresses, benchmarks, evaluates, or
    only partially solves it - the reviewer needs the note, not a silent
    auto-reject."""
    if not matches:
        return status, None
    best = max(matches, key=lambda m: m.similarity)
    note = (
        f'"{best.paper_title}" has a contribution/results claim that closely resembles this pattern '
        f"(similarity {best.similarity:.2f}) — it may already address, evaluate, or partially solve this; "
        f"verify before treating it as fully open."
    )
    return _DOWNGRADE[status], note
