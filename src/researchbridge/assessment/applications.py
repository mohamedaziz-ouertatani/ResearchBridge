"""Evidence-grounded application discovery (blueprint Sec 2A/33).

Deliberately narrow, like assess_research_gap: only surfaces applications
explicitly stated by a relevant retrieved paper's own "applications"
extraction claim (Sec 28/29's ninth extraction field) - never synthesizes
a novel application by combining or clustering across papers. That
further interpretive leap is "Potential Product/Technology Opportunities"
(Sec 49) - a separate, not-yet-built report field.

Two independent gates protect against weak/false "applications" - see
docs/superpowers/specs/2026-09-02-applications-evidence-grounding-design.md:

Gate 1 (extraction/validation.py::validate_claim_type, "applications"
branch) runs at extraction time on a single candidate sentence with no
paper context - it rejects text that doesn't read as genuine deployment/
use-case language at all (method sentences, generic motivation, the bare
"software application" noun sense).

Gate 2 (this module, _restates_own_task below) runs at assessment time,
WITH paper context: even a claim that passed Gate 1 can still be a near-
total paraphrase of the SAME paper's own problem/method/main_contribution/
results claim ("useful for predicting performance level of students" when
the paper's own task IS predicting student performance). Gate 1 cannot
detect this - it has no way to know what the paper's own task is. This
reuses the TECHNIQUE gaps/detect.py::_own_contribution_overlaps() uses
(max cosine similarity to the same paper's own claims), applied to a
different question and a different, currently-provisional threshold -
duplicated here deliberately rather than imported, since the two modules'
surrounding logic and claim-type sets differ enough that sharing code
across two call sites would add more indirection than it saves.

Relevance-gated the same way as assess_research_gap: a retrieved paper
only contributes if its distance is within RELEVANCE_DISTANCE (novelty
.py's calibrated FAR_DISTANCE), so an off-topic paper's applications never
leak into the report.

Pure function, no DB access: like novelty.py/coverage.py, operates
entirely on the (paper_id, title, distance, claims) tuples
build_assessment() has already assembled - this module used to run its
own claim_type=="applications" query, which was redundant once every
relevant paper's full claim set was already in memory for other
assess_* functions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.embedding.base import Embedder

ClaimRecord = tuple[str, str, uuid.UUID]  # (claim_type, text, evidence_id)
PaperWithClaims = tuple[uuid.UUID, str, float, list[ClaimRecord]]  # (paper_id, title, distance, claims)

# The claim types that describe what a paper's own research task IS - an
# "applications" claim whose text is a near-total paraphrase of one of
# these is restating the paper's own work, not naming an external
# deployment. Broader than gaps/detect.py's CONTRIBUTION_CLAIM_TYPES
# ("main_contribution", "results" only): gap's own-contribution-overlap
# answers a narrower question (does the paper's OWN CONTRIBUTION already
# solve the gap it named); applications' failure mode is a claim
# restating the task under ANY of its guises, not just the contribution.
_TASK_CLAIM_TYPES = ("problem", "method", "main_contribution", "results")

# Checked against the real corpus/embedder (all-MiniLM-L6-v2), not just
# guessed: of 348 persisted "applications" claims that still pass the
# strengthened Gate 1, 587 (application, same-paper task) pairs were
# compared. 60/587 sit at or above 0.92, and inspection showed these are
# overwhelmingly EXACT-TEXT duplicates (sim=1.000) - the same sentence
# extracted twice under two different claim_types across separate
# extraction runs, not just similar phrasing. The one genuinely high
# (but non-identical) real pair found close to the boundary - "...can be
# effectively used to predict option prices...and could be useful for
# OPTIONS TRADERS in making informed decisions" (sim=0.899) against a task
# claim naming no trader/decision language at all - correctly stays
# ACCEPTED at 0.92: it names a real actor and downstream action the task
# claim doesn't have, exactly the pattern this check must not reject.
# Lowering the threshold to something like 0.85 to more aggressively catch
# restatement would have incorrectly rejected that genuine case too, so
# 0.92 was kept rather than tuned down.
#
# Two real residual false positives WERE found in the 0.70-0.92 band
# ("...useful for predicting future trips for A GIVEN USER" against a
# trip-prediction task claim, sim=0.848; "...applied in QPE" - QPE being
# the method's own acronym - against its own definition, sim=0.848) - but
# both trace to extraction/validation.py's _QUALIFYING_CONTEXT_RE fallback
# being too permissive (matching "for a given user"/"in QPE" as if they
# named genuine external context), not to this threshold. Lowering this
# threshold would not have fixed either case cleanly without also risking
# the genuine 0.899 case above. Left as a known Gate 1 refinement
# candidate for a future pass, not addressed here - see
# docs/superpowers/plans/2026-09-02-applications-evidence-grounding.md.
OWN_TASK_OVERLAP_THRESHOLD = 0.92


@dataclass
class ApplicationRecord:
    application: str
    source_paper: str
    paper_id: uuid.UUID
    evidence_id: uuid.UUID


@dataclass
class ApplicationsResult:
    applications: list[ApplicationRecord] | None
    evidence_ids: list[uuid.UUID]
    status: str = "not_assessed"  # "not_assessed" | "no_evidence" | "found"


_NOT_ASSESSED_RESULT = ApplicationsResult(applications=None, evidence_ids=[], status="not_assessed")
_NO_EVIDENCE_RESULT = ApplicationsResult(applications=[], evidence_ids=[], status="no_evidence")


def assess_applications(papers_with_claims: list[PaperWithClaims], embedder: Embedder) -> ApplicationsResult:
    """papers_with_claims: every retrieved paper's id, title, distance, and
    already-extracted non-stub claims, nearest-first (the shape
    build_assessment() already assembles before calling this)."""
    relevant = [p for p in papers_with_claims if p[2] <= RELEVANCE_DISTANCE]
    if not relevant:
        return _NOT_ASSESSED_RESULT

    app_candidates = [
        (paper_id, title, text, evidence_id)
        for paper_id, title, _distance, claims in relevant
        for claim_type, text, evidence_id in claims
        if claim_type == "applications"
    ]
    if not app_candidates:
        return _NO_EVIDENCE_RESULT

    task_texts_by_paper: dict[uuid.UUID, list[str]] = {}
    for paper_id, _title, _distance, claims in relevant:
        for claim_type, text, _evidence_id in claims:
            if claim_type in _TASK_CLAIM_TYPES:
                task_texts_by_paper.setdefault(paper_id, []).append(text)

    app_texts = [c[2] for c in app_candidates]
    all_task_texts = [t for texts in task_texts_by_paper.values() for t in texts]
    vectors = embedder.embed_texts(app_texts + all_task_texts)
    app_vectors = vectors[: len(app_texts)]
    task_vectors = vectors[len(app_texts) :]
    vector_by_task_text: dict[str, list[float]] = dict(zip(all_task_texts, task_vectors, strict=True))

    applications: list[ApplicationRecord] = []
    evidence_ids: list[uuid.UUID] = []
    for (paper_id, title, text, evidence_id), app_vector in zip(app_candidates, app_vectors, strict=True):
        own_task_texts = task_texts_by_paper.get(paper_id, [])
        if own_task_texts and _restates_own_task(app_vector, own_task_texts, vector_by_task_text):
            continue
        applications.append(
            ApplicationRecord(application=text, source_paper=title, paper_id=paper_id, evidence_id=evidence_id)
        )
        evidence_ids.append(evidence_id)

    if not applications:
        return _NO_EVIDENCE_RESULT
    return ApplicationsResult(applications=applications, evidence_ids=evidence_ids, status="found")


def _restates_own_task(
    app_vector: list[float], own_task_texts: list[str], vector_by_task_text: dict[str, list[float]]
) -> bool:
    own_vectors = [vector_by_task_text[t] for t in own_task_texts]
    overlap = max(_cosine(app_vector, v) for v in own_vectors)
    return overlap >= OWN_TASK_OVERLAP_THRESHOLD


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
