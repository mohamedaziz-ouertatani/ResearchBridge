"""Input-specific research-gap assessment (blueprint Sec 2A/32).

Scoped to one ResearchAssessment's own retrieved-paper neighborhood, not
the whole corpus - deliberately distinct from gaps/detect.py's per-seed-
paper corpus similarity search. This assessment already has its own
retrieved papers (with distances) from search_by_text(), so there's no
seed paper to expand a neighborhood from.

Relevance gate: a retrieved paper only participates if its distance is
within RELEVANCE_DISTANCE (reusing novelty.py's own calibrated
FAR_DISTANCE - the same "genuinely related vs. off-topic" boundary,
rather than inventing a second threshold). Checked live against the real
corpus: without this gate, an idea unrelated to anything in the corpus
(e.g. "cats playing chess with quantum computers") still surfaced an
explicit gap from whichever barely-related paper happened to rank in the
top-k - technically "evidence", but not actually relevant evidence. The
gate turns that into NULL/NOT_ASSESSED instead, per Sec 22.

Priority order, each labeled distinctly so a human reader never confuses
inference with a stated fact (Sec 34):

1. An already-reviewed ("approved") candidate_gaps row whose seed_paper_id
   is a relevant retrieved paper - reused outright. Still labeled as
   inference (candidate_gaps.gap_type is always "inference").
2. An explicit, author-stated research_gap claim on a relevant retrieved
   paper (Sec 32's "Explicit gaps" - the research_gap extraction field).
   The most direct source; never described as inferred. Checked
   nearest-first, not via an unordered SQL IN query - the DB does not
   preserve retrieval relevance order on its own.
3. A fresh cross-paper cluster over the relevant neighborhood's
   limitations/research_gap claims, reusing gaps/cluster.py's
   find_recurring_patterns directly (not gaps/detect.py's seed-paper
   machinery, since the neighborhood is already this assessment's own
   retrieval). Explicitly labeled as inference, matching gaps/detect.py's
   own phrasing.
4. Nothing found -> NULL/NOT_ASSESSED. Insufficient (or irrelevant)
   evidence stays insufficient; nothing here is ever fabricated.

is_closely_grounded: a second, stricter signal alongside the report text,
added after real-corpus testing showed recommendation.py crediting "a gap
was found" as equally strong evidence whether the source paper was a
close match or merely inside the loose RELEVANCE_DISTANCE band - the same
kind of over-crediting that feasibility.py's threshold had to be tightened
for. True only when at least one paper backing the result clears
novelty.py's tighter NEAR_DISTANCE (0.35), not just RELEVANCE_DISTANCE
(0.65). The report field (`text`) is unaffected either way - it still
surfaces within the looser band - this only marks whether recommendation
should treat it as a strong signal.

is_strongly_stated: a third signal, orthogonal to is_closely_grounded -
distance measures whether the SOURCE PAPER is a close topical match;
this measures whether the GAP TEXT ITSELF is a specific, substantive
statement rather than generic boilerplate. Found live-testing real ideas
(2026-09-04): a fraud-detection assessment reported "HIGH PRIORITY / high
confidence" whose gap evidence was "Extensive experiments yield empirical
insights into current methods' limitations and suggest promising avenues
for future research" - textbook generic "future research" boilerplate,
extraction/validation.py's own weak tier (a source paper close enough to
be topically relevant, but a sentence with no specific content). Confidence
was "high" only because assessed_count treated "a value exists" as
equivalent to "strong evidence" - the same class of bug is_closely_grounded
already fixed for the distance axis. True for source="input_specific" only
when the underlying claim's validation_tier is "strong" (never "weak" -
see extraction/validation.py's own docstring on why the weak tier is
"ambiguous" by design); always True for "reused_candidate_gap" (a human
already reviewed and approved it); for the inferred cross-paper cluster
path, True only when the cluster's own tier is "strong_gap" - see `tier`
below and gaps/cluster.py's GapCluster.tier docstring.

tier: a fourth signal, classifying WHICH KIND of gap this is rather than
how strong it is - "strong_gap" | "potential_gap" | "known_limitation" |
None. Found live-testing a real fraud-detection assessment (2026-09-04):
the inferred cross-paper path always reported "Recurring pattern across N
related papers" and always set is_strongly_stated=True for ANY cluster
that cleared DEFAULT_SIMILARITY_THRESHOLD, whether the N papers actually
described the same specific unresolved problem or merely a broader shared
theme - e.g. one paper on FL privacy/communication/architecture, one on FL
security/poisoning, one on FL poisoning/right-to-be-forgotten: genuinely
related, but not the same specific problem, yet worded and scored
identically to a cluster where all three papers name the exact same issue.
- "strong_gap": reused_candidate_gap (human-reviewed) always; the inferred
  cluster when GapCluster.tier == "strong_gap" (see its own docstring for
  the lexical-overlap calibration).
- "potential_gap": the inferred cluster when GapCluster.tier ==
  "potential_gap" - still surfaced (Sec 22 - insufficient is not the same
  as absent, and a broader recurring theme is real signal), just not
  conflated with "same problem" wording or is_strongly_stated=True.
- "known_limitation": an explicit, single-paper research_gap claim - by
  construction there is exactly one paper behind it, so it is a genuine,
  author-stated limitation, but has no cross-paper convergence to speak of.
- None: no gap found/assessed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.assessment.novelty import NEAR_DISTANCE as CLOSE_DISTANCE
from researchbridge.db.models import CandidateGap, CandidateGapEvidence, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.base import Embedder
from researchbridge.gaps.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    RELEVANT_CLAIM_TYPES,
    ClaimRecord,
    find_recurring_patterns,
)

PaperWithDistance = tuple[uuid.UUID, float]


@dataclass
class GapAssessmentResult:
    source: str | None  # "reused_candidate_gap" | "input_specific" | None
    text: str | None
    candidate_gap_id: uuid.UUID | None
    evidence_ids: list[uuid.UUID]
    is_closely_grounded: bool = False
    is_strongly_stated: bool = False
    status: str = "not_assessed"  # "not_assessed" | "not_found" | "found"
    tier: str | None = None  # "strong_gap" | "potential_gap" | "known_limitation" | None


_NOT_ASSESSED_RESULT = GapAssessmentResult(
    source=None, text=None, candidate_gap_id=None, evidence_ids=[], status="not_assessed"
)
_NOT_FOUND_RESULT = GapAssessmentResult(
    source=None, text=None, candidate_gap_id=None, evidence_ids=[], status="not_found"
)


def assess_research_gap(
    session: Session,
    papers_by_distance: list[PaperWithDistance],
    embedder: Embedder,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> GapAssessmentResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns)."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    if not relevant_paper_ids:
        return _NOT_ASSESSED_RESULT

    result = _reuse_approved_candidate_gap(session, relevant_paper_ids)
    if result is None:
        result = _explicit_research_gap_claim(session, relevant_paper_ids)
    if result is None:
        result = _inferred_cross_paper_gap(session, relevant_paper_ids, embedder, min_cluster_size, similarity_threshold)
    if result is None:
        return _NOT_FOUND_RESULT

    result.status = "found"
    distance_by_paper = dict(papers_by_distance)
    result.is_closely_grounded = _any_evidence_paper_is_close(session, result.evidence_ids, distance_by_paper)
    return result


def _any_evidence_paper_is_close(
    session: Session, evidence_ids: list[uuid.UUID], distance_by_paper: dict[uuid.UUID, float]
) -> bool:
    if not evidence_ids:
        return False
    paper_ids = session.execute(select(Evidence.paper_id).where(Evidence.id.in_(evidence_ids))).scalars().all()
    return any(distance_by_paper.get(paper_id, RELEVANCE_DISTANCE + 1) <= CLOSE_DISTANCE for paper_id in paper_ids)


def _reuse_approved_candidate_gap(
    session: Session, relevant_paper_ids: list[uuid.UUID]
) -> GapAssessmentResult | None:
    gap = session.execute(
        select(CandidateGap)
        .where(CandidateGap.seed_paper_id.in_(relevant_paper_ids), CandidateGap.status == "approved")
        .order_by(CandidateGap.contributing_paper_count.desc())
    ).scalars().first()
    if gap is None:
        return None

    evidence_ids = list(
        session.execute(
            select(CandidateGapEvidence.evidence_id).where(CandidateGapEvidence.candidate_gap_id == gap.id)
        ).scalars()
    )
    return GapAssessmentResult(
        source="reused_candidate_gap",
        text=gap.observation,
        candidate_gap_id=gap.id,
        evidence_ids=evidence_ids,
        # a human already reviewed and approved this - not re-judged on
        # wording the way a fresh extracted claim is
        is_strongly_stated=True,
        tier="strong_gap",
    )


def _explicit_research_gap_claim(
    session: Session, relevant_paper_ids: list[uuid.UUID]
) -> GapAssessmentResult | None:
    """Checks papers in the given order (nearest-first, per the caller's own
    retrieval ranking) and returns the first explicit claim found - NOT an
    arbitrary one. An unordered `paper_id IN (...)` query would let the DB
    return any matching row first, regardless of how relevant that paper
    actually is to the input; that defeats the point of scoping this to
    the assessment's own neighborhood."""
    rows = session.execute(
        select(
            ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title,
            ExtractedClaim.validation_tier,
        )
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "research_gap",
            Evidence.extraction_method != "stub",
        )
    ).all()
    by_paper_id = {row.paper_id: (row.text, row.evidence_id, row.title, row.validation_tier) for row in rows}

    for paper_id in relevant_paper_ids:
        if paper_id in by_paper_id:
            text, evidence_id, paper_title, validation_tier = by_paper_id[paper_id]
            return GapAssessmentResult(
                source="input_specific",
                text=f'Explicitly stated in "{paper_title}": "{text}"',
                candidate_gap_id=None,
                evidence_ids=[evidence_id],
                # "strong" tier means unambiguous gap language ("remains an
                # open problem"); "weak" means boilerplate that could just as
                # easily be generic filler ("future research") - see
                # extraction/validation.py's own strong/weak tier docstring.
                # Never fabricate a strength judgment the extractor didn't
                # already make.
                is_strongly_stated=(validation_tier == "strong"),
                # exactly one paper behind this claim - genuine, but no
                # cross-paper convergence by construction
                tier="known_limitation",
            )
    return None


def _inferred_cross_paper_gap(
    session: Session,
    relevant_paper_ids: list[uuid.UUID],
    embedder: Embedder,
    min_cluster_size: int,
    similarity_threshold: float,
) -> GapAssessmentResult | None:
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.evidence_id, ExtractedClaim.text)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type.in_(RELEVANT_CLAIM_TYPES),
            Evidence.extraction_method != "stub",
        )
    ).all()
    claims = [ClaimRecord(paper_id=paper_id, evidence_id=evidence_id, text=text) for paper_id, evidence_id, text in rows]

    clusters = find_recurring_patterns(claims, embedder, min_cluster_size, similarity_threshold)
    if not clusters:
        return None

    top = clusters[0]  # already sorted by contributing_paper_count, descending
    if top.tier == "strong_gap":
        # every pairwise member shares real vocabulary - likely the same
        # specific problem, not just a broader shared theme
        text = (
            f"Recurring pattern across {top.contributing_paper_count} related papers, describing "
            f'the same specific issue (inference, not stated by any single author): "{top.representative_text}"'
        )
    else:
        text = (
            f"{top.contributing_paper_count} related papers share a broader recurring theme, though not "
            f'necessarily the same specific unresolved problem (inference, not stated by any single '
            f'author): "{top.representative_text}"'
        )
    evidence_ids = [member.evidence_id for member in top.members]
    return GapAssessmentResult(
        source="input_specific",
        text=text,
        candidate_gap_id=None,
        evidence_ids=evidence_ids,
        # only the tight, same-specific-problem tier counts as a strong
        # signal - a broader shared theme is real but weaker evidence, see
        # gaps/cluster.py's GapCluster.tier docstring
        is_strongly_stated=(top.tier == "strong_gap"),
        tier=top.tier,
    )
