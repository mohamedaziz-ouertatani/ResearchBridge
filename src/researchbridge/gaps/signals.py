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
# cover what it just named as a gap). Starting value, not yet calibrated
# against the real corpus embedder - see Task 13's calibration step, which
# must update this comment with the measured distribution before treating
# the value as final.
OWN_CONTRIBUTION_OVERLAP_THRESHOLD = 0.5

# Same idea, cross-paper: how similar a cluster's representative gap text
# needs to be to some OTHER paper's main_contribution/results claim before
# that counts as an "addressing signal" worth a reviewer note and a one-
# level downgrade (never an exclusion - see apply_addressing_downgrade).
# Same calibration caveat as above.
ADDRESSING_SIMILARITY_THRESHOLD = 0.5


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
