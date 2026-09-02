# Applications Evidence Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the assessment's "Potential Applications" section from surfacing task-restatements and homonym false positives, without weakening genuine deployment/use-case claims, and without any LLM, cross-paper synthesis, or schema migration.

**Architecture:** A two-gate design. Gate 1 (`extraction/validation.py`, extraction time, no paper context) requires genuine deployment-framing language with a concrete, named external complement, and excludes the bare "software application" noun sense. Gate 2 (`assessment/applications.py`, assessment time, same-paper context) rejects an application claim that's a near-total paraphrase of the *same paper's* own problem/method/contribution/results claim — a technique ported from `gaps/detect.py::_own_contribution_overlaps`, with its own provisional threshold. `assess_applications` becomes a pure function (drops its DB dependency), consuming the paper/claims bundle `build_assessment` already assembles.

**Tech Stack:** Python 3.12, regex (`re`), the existing `Embedder` protocol (`embed_texts`) — no new dependency.

**Spec:** `docs/superpowers/specs/2026-09-02-applications-evidence-grounding-design.md`

## Global Constraints

- Fully deterministic/extractive — no LLM anywhere in this change.
- Verbatim evidence, real `evidence_id` provenance — unchanged.
- Independent per-paper provenance — never merge/synthesize across papers.
- `None` (no relevant papers) vs. `[]` (relevant papers, no admissible application) semantics — preserved exactly.
- No changes to `coverage.py` / `DimensionCoverage`.
- No cross-paper application clustering or synthesis.
- No `tier="weak"` accepted state — hedged/generic language is rejected outright.
- No database migration — `ExtractedClaim.validation_tier` already exists and is currently unused for `applications`.
- No `schemas.py` / `AssessmentReport.tsx` changes in this plan (deferred, out of scope per spec).

---

## Task 1: Strengthen Gate 1 — `extraction/validation.py`'s `applications` branch

**Files:**
- Modify: `src/researchbridge/extraction/validation.py`
- Test: `tests/test_extraction_validation.py`

**Interfaces:**
- Produces: `validate_claim_type("applications", text)` now returns `ValidationResult(is_valid=True, tier="strong")` on acceptance (previously always `tier=None`), and a more specific rejection reason on failure. No signature change — same `validate_claim_type(claim_type: str, text: str) -> ValidationResult`.

This task only touches the `applications`-specific regex and the one `if claim_type == "applications":` branch — every other claim type's validation logic is untouched.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_extraction_validation.py`, right after the existing `test_generic_usefulness_statement_is_not_enough_for_applications` (keep the four existing applications tests in the file as-is — they must still pass unmodified):

```python
def test_task_restatement_dressed_as_usefulness_is_rejected() -> None:
    # the reported bug: no actor, no institution, no downstream action -
    # just the paper's own predictive task restated as a gerund
    text = (
        "In this paper will deliberate various techniques of data mining "
        "which are useful for predicting performance level of students."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is False
    assert result.reason


def test_software_applications_noun_sense_is_rejected() -> None:
    # the reported bug: "applications" meaning computer programs, not
    # "applications of this research" - a lexical homonym, not a weak signal
    text = (
        "These are aided by the automation of many procedures involved in "
        "typical student activities, which manage huge amounts of information "
        "collected through software applications for technology-oriented learning."
    )
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_application_naming_an_actor_is_accepted() -> None:
    text = "Deployed by banks to flag suspicious transactions for manual review by compliance teams."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_application_naming_a_downstream_action_is_accepted() -> None:
    text = "Can be used by dermatologists to prioritize which patients need urgent biopsy."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_application_naming_an_external_setting_is_accepted() -> None:
    # no actor/action keyword, but a genuine "in X" domain qualifier beyond
    # the bare task object - must still be accepted
    text = "Can be applied to accelerate drug discovery pipelines in pharmaceutical R&D."
    result = validate_claim_type("applications", text)

    assert result.is_valid is True


def test_bare_predictive_task_with_no_qualifier_is_rejected() -> None:
    # same shape as the reported bug but phrased with "applied to" instead
    # of "useful for" - must still be rejected, since the complement names
    # nothing beyond the task's own object
    text = "This approach can be applied to predicting customer churn."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_method_language_with_used_to_is_still_rejected() -> None:
    # "used" is a recognized deployment verb, but "used to train" is method
    # language, not a deployment claim - the complement must still clear
    # the actor/action/qualifying-context requirement
    text = "The dataset was used to train the model on benchmark image data."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False


def test_vague_qualifier_still_rejected_even_with_a_trailing_in_phrase() -> None:
    # "in general" must not be mistaken for a genuine domain qualifier
    text = "These findings could be useful for future studies in general."
    result = validate_claim_type("applications", text)

    assert result.is_valid is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction_validation.py -v`
Expected: the four existing applications tests still PASS (they exercise cases the rewrite must keep working); the eight new tests FAIL — most will fail with `AttributeError`/assertion mismatches since `_has_application_signal` doesn't yet implement this logic, and `test_application_naming_an_actor_is_accepted` will additionally fail on `result.tier == "strong"` since `tier` isn't set today.

- [ ] **Step 3: Rewrite the `applications` section of `extraction/validation.py`**

Replace the existing `_APPLICATION_LANGUAGE_RE` / `_VAGUE_USEFULNESS_RE` block (currently around lines 97-120) with:

```python
# Real application/use-context language - Gate 1 of a two-gate design (see
# docs/superpowers/specs/2026-09-02-applications-evidence-grounding-design.md).
# This gate asks a narrow, context-free question: does this sentence, read
# on its own, describe genuine deployment/use-case language with a
# concrete, named external complement? It deliberately does NOT try to
# detect whether the claim restates the SAME PAPER's own task - that
# requires comparing against the paper's other claims, which this
# function has no access to (it validates one candidate sentence at a
# time). That check is Gate 2, in assessment/applications.py, at
# assessment time, where the paper's other claims are available.
#
# Two real production false positives motivated the redesign of the old,
# single-regex version of this check:
#   1. "...useful for predicting performance level of students." - the
#      old _APPLICATION_LANGUAGE_RE's `useful (to|for|in) [a-z]` clause
#      accepted ANY following word, so a bare restatement of the paper's
#      own predictive task (no actor, no institution, no downstream
#      action, nothing beyond the task's own object) passed.
#   2. "...software applications for technology-oriented learning." - the
#      old regex's `applications? (such as|include|in|to|for)` matched the
#      literal substring "applications for" regardless of whether
#      "application(s)" meant "use-case" or "computer program."

# Strip the ordinary "software/mobile/web/computer application(s)" noun
# sense before matching anything else, so it can never itself satisfy the
# "applications... for/to/in" clause below.
_TECH_NOUN_APPLICATION_RE = re.compile(
    r"\b(software|mobile|web|desktop|computer|online)\s+applications?\b", re.IGNORECASE
)

# Each alternative captures its own complement (the text after the
# deployment verb) into a distinctly-named group, so the code below can
# find whichever one matched and inspect it.
_DEPLOYMENT_CLAUSE_RE = re.compile(
    r"\b(?:can|could) be (?:applied|used|deployed)\s+(?:to|for|in)\s+(?P<c1>[^.;]+)"
    r"|\bis applicable\s+(?:to|in)\s+(?P<c2>[^.;]+)"
    r"|\bapplications?\s+(?:such as|include|in|to|for)\s+(?P<c3>[^.;]+)"
    r"|\breal-world applications?\s+(?:in|for)\s+(?P<c4>[^.;]+)"
    r"|\bdeployed\s+(?:in|to|for|by)\s+(?P<c5>[^.;]+)"
    r"|\bused\s+(?:in|for|to|by)\s+(?P<c6>[^.;]+)"
    r"|\b(?:applied|applicable|targets?|targeting)\s+(?:to|for|in)\s+(?P<c7>[^.;]+)"
    r"|\buseful\s+(?:to|for|in)\s+(?P<c8>[^.;]+)",
    re.IGNORECASE,
)

# Named human/institutional actors or deployment settings - if the
# deployment verb's complement names one of these, the claim is naming
# something beyond the paper's own task.
_ACTOR_SETTING_RE = re.compile(
    r"\buniversit(y|ies)\b|\bschools?\b|\bhospitals?\b|\bclinics?\b|\bclinicians?\b"
    r"|\bphysicians?\b|\bdoctors?\b|\bnurses?\b|\bbanks?\b|\bbanking\b"
    r"|\bfinancial institutions?\b|\bcompliance teams?\b|\bregulators?\b"
    r"|\bpolicymakers?\b|\binstructors?\b|\bteachers?\b|\beducators?\b"
    r"|\badvisors?\b|\bcounselors?\b|\bpractitioners?\b|\bindustry\b"
    r"|\borganizations?\b|\bcompanies\b|\bbusinesses?\b|\benterprises?\b"
    r"|\bgovernments?\b|\bagenc(y|ies)\b|\bdecision[- ]makers?\b|\bstakeholders?\b"
    r"|\b\w+(ologists?|icians?)\b"
    r"|\bin (clinical|industrial|educational|practical|real-world) (practice|settings?|use|contexts?)\b"
    r"|\bat scale\b|\bin the field\b|\bin practice\b",
    re.IGNORECASE,
)

# Downstream actions distinct from the paper's own predictive/detection/
# classification verb - a human or institutional response taken as a
# RESULT of the system's output, not the system's own computation.
_DOWNSTREAM_ACTION_RE = re.compile(
    r"\binterventions?\b|\btriage\b|\bmanual review\b|\bcounsel(l)?ing\b"
    r"|\btreatment plan(ning)?\b|\bresource allocation\b|\bpolicy( ?making)?\b"
    r"|\bremediation\b|\bprioriti[sz](e|ation|ing)\b|\bdecision support\b"
    r"|\brisk mitigation\b|\bearly (intervention|warning)\b|\bscreening\b"
    r"|\breferrals?\b|\bflagg?ing\b|\balert(ing)?\b",
    re.IGNORECASE,
)

# A generic structural fallback: the complement contains ITS OWN
# qualifying "in X"/"for X"/"at X"/"by X" phrase beyond the deployment
# verb's own preposition - e.g. "fraud detection IN financial
# transactions", "drug discovery pipelines IN pharmaceutical R&D". Deliberately
# excludes "on" (method/training language routinely says "trained on X").
_QUALIFYING_CONTEXT_RE = re.compile(r"\b(in|for|at|by|within|across|among)\s+[a-z]", re.IGNORECASE)

# Known-vague qualifiers that would otherwise satisfy _QUALIFYING_CONTEXT_RE
# without naming anything concrete - "in general" contains "in general[a-z]"
# but names no real context.
_VAGUE_QUALIFIER_RE = re.compile(
    r"\bin general\b|\bfor future (work|studies|research)\b|\bin the future\b"
    r"|\bfor general purposes\b|\bfor further research\b|\bin future\b",
    re.IGNORECASE,
)


def _has_application_signal(text: str) -> bool:
    masked = _TECH_NOUN_APPLICATION_RE.sub(" ", text)
    match = _DEPLOYMENT_CLAUSE_RE.search(masked)
    if not match:
        return False
    complement = next(g for g in match.groupdict().values() if g)
    if _ACTOR_SETTING_RE.search(complement) or _DOWNSTREAM_ACTION_RE.search(complement):
        return True
    if _VAGUE_QUALIFIER_RE.search(complement):
        return False
    return bool(_QUALIFYING_CONTEXT_RE.search(complement))
```

Then update the `applications` branch inside `validate_claim_type` (currently around line 189):

```python
    if claim_type == "applications":
        if not _has_application_signal(text):
            return ValidationResult(
                False,
                "no concrete deployment context (actor, institution, downstream action, or "
                "named external setting) found; reads as method, result, restated task, or "
                "generic/vague text",
            )
        return ValidationResult(True, tier="strong")
```

Delete the now-unused `_APPLICATION_LANGUAGE_RE` and `_VAGUE_USEFULNESS_RE` regexes and the `_has_application_signal` function's old body (both replaced above). Update the module's top docstring paragraph that currently says "research_gap, applications, limitations, results, method, problem" have "no natural lexicon to check against" for unvalidated types — no change needed there, it already correctly excludes `applications` from that list (it's validated). Update the `ValidationResult.tier` docstring (currently says `""strong" or "weak" for an accepted research_gap claim... None for every other claim_type"`) to:

```python
    tier: str | None = None
    """"strong" or "weak" for an accepted research_gap claim (which regex
    tier matched); "strong" for an accepted applications claim (there is
    no "weak" applications tier - hedged/generic language is rejected
    outright, not accepted-but-flagged); None for every other claim_type,
    and None for any rejected claim. Persisted downstream
    (extracted_claims.validation_tier)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction_validation.py -v`
Expected: PASS — all 4 pre-existing applications tests, all 8 new tests, and every other test in the file (research_gap/results/limitations/method/problem branches are untouched).

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/validation.py tests/test_extraction_validation.py
git commit -m "feat(extraction): require named deployment context for applications claims"
```

---

## Task 2: Add Gate 2 (own-task-overlap) and make `assess_applications` a pure function

**Files:**
- Modify: `src/researchbridge/assessment/applications.py`
- Test: `tests/test_assessment_applications.py` (full rewrite — the pure-function signature removes the need for a DB session in every test)

**Interfaces:**
- Consumes: `Embedder.embed_texts(texts: list[str]) -> list[list[float]]` (existing protocol).
- Produces: `ClaimRecord = tuple[str, str, uuid.UUID]` (claim_type, text, evidence_id), `PaperWithClaims = tuple[uuid.UUID, str, float, list[ClaimRecord]]` (paper_id, title, distance, claims), `assess_applications(papers_with_claims: list[PaperWithClaims], embedder: Embedder) -> ApplicationsResult`. **Breaking signature change** from today's `assess_applications(session: Session, papers_by_distance: list[tuple[uuid.UUID, float]]) -> ApplicationsResult` — Task 3 updates the one call site.

- [ ] **Step 1: Write the failing tests (full replacement of the test file)**

Replace the entire contents of `tests/test_assessment_applications.py`:

```python
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.applications import assess_applications

NEAR = 0.1
FAR = 0.9

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake used elsewhere in this suite (test_assessment_gap.py,
    test_assessment_coverage.py): cosine similarity of two texts is
    exactly their fraction of shared distinct words."""

    model_name: str = "word-overlap-fake"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vocab = sorted({w for t in texts for w in _WORD.findall(t.lower())})
        vectors = []
        for t in texts:
            words = set(_WORD.findall(t.lower()))
            raw = [1.0 if w in words else 0.0 for w in vocab]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


@pytest.fixture()
def embedder() -> WordOverlapEmbedder:
    return WordOverlapEmbedder()


def _claim(claim_type: str, text: str) -> tuple[str, str, uuid.UUID]:
    return (claim_type, text, uuid.uuid4())


def test_returns_none_for_empty_neighborhood(embedder) -> None:
    result = assess_applications([], embedder)
    assert result.applications is None
    assert result.evidence_ids == []


def test_returns_none_when_no_relevant_papers(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", FAR, [_claim("applications", "fraud detection")])], embedder
    )
    assert result.applications is None


def test_returns_empty_list_when_no_applications_claims_exist(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("limitations", "some limitation")])], embedder
    )
    assert result.applications == []


def test_collects_application_from_a_relevant_paper(embedder) -> None:
    paper_id = uuid.uuid4()
    app_claim = _claim("applications", "real-time payment fraud screening")

    result = assess_applications([(paper_id, "Fraud Paper", NEAR, [app_claim])], embedder)

    assert result.applications is not None
    assert len(result.applications) == 1
    app = result.applications[0]
    assert app.application == "real-time payment fraud screening"
    assert app.source_paper == "Fraud Paper"
    assert app.paper_id == paper_id
    assert result.evidence_ids == [app_claim[2]]


def test_collects_applications_from_multiple_relevant_papers_nearest_first(embedder) -> None:
    near_id, far_id = uuid.uuid4(), uuid.uuid4()
    near_claim = _claim("applications", "near application")
    far_claim = _claim("applications", "far application")

    result = assess_applications(
        [(near_id, "Near Paper", 0.1, [near_claim]), (far_id, "Far Paper", 0.4, [far_claim])], embedder
    )

    assert [a.application for a in result.applications] == ["near application", "far application"]
    assert result.evidence_ids == [near_claim[2], far_claim[2]]


def test_excludes_too_distant_papers(embedder) -> None:
    near_id, far_id = uuid.uuid4(), uuid.uuid4()
    near_claim = _claim("applications", "a real application")
    far_claim = _claim("applications", "an irrelevant application")

    result = assess_applications(
        [(near_id, "Near", NEAR, [near_claim]), (far_id, "Far", FAR, [far_claim])], embedder
    )

    assert len(result.applications) == 1
    assert result.applications[0].application == "a real application"
    assert result.evidence_ids == [near_claim[2]]


def test_status_not_assessed_when_no_relevant_papers(embedder) -> None:
    result = assess_applications([], embedder)
    assert result.status == "not_assessed"
    assert result.applications is None


def test_status_no_evidence_when_relevant_papers_have_no_application_claim(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("method", "an unrelated method")])], embedder
    )
    assert result.status == "no_evidence"
    assert result.applications == []


def test_status_found_when_an_application_claim_exists(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("applications", "used in real-time fraud monitoring")])], embedder
    )
    assert result.status == "found"
    assert result.applications is not None
    assert len(result.applications) == 1


def test_own_task_overlap_does_not_reject_a_genuinely_external_application(embedder) -> None:
    # your worked example: high topical relatedness, but the application
    # names an actor (universities) and a downstream action (intervention)
    # the task claim doesn't have
    task = _claim(
        "method", "predicts which students are at risk of failing using grades and attendance data"
    )
    app = _claim(
        "applications",
        "support universities in identifying students at risk of failure early enough for intervention",
    )
    paper_id = uuid.uuid4()

    result = assess_applications([(paper_id, "Student Risk Paper", NEAR, [task, app])], embedder)

    assert result.status == "found"
    assert len(result.applications) == 1
    assert result.applications[0].application == app[1]


def test_own_task_overlap_rejects_a_near_total_paraphrase_of_the_papers_own_task(embedder) -> None:
    # identical text on both sides is a deliberate stand-in for a real
    # near-paraphrase: the coarse word-overlap fake embedder needs near-
    # identical wording to clear OWN_TASK_OVERLAP_THRESHOLD, whereas a
    # real embedder would already show high-but-not-identical similarity
    # for a genuine paraphrase like "university-level performance
    # prediction" vs. "predicting student academic performance" - this
    # test exercises the same rejection code path unambiguously
    restated_text = "predicts which students are at risk of failing using grades and attendance data"
    task = _claim("method", restated_text)
    app = _claim("applications", restated_text)
    paper_id = uuid.uuid4()

    result = assess_applications([(paper_id, "Student Risk Paper", NEAR, [task, app])], embedder)

    assert result.status == "no_evidence"
    assert result.applications == []


def test_own_task_overlap_only_compares_against_the_same_paper(embedder) -> None:
    # paper A's application claim happens to closely paraphrase paper B's
    # task, not its own - must not be rejected for a cross-paper match
    paper_a, paper_b = uuid.uuid4(), uuid.uuid4()
    shared_text = "predicts which students are at risk of failing using grades and attendance data"
    task_b = _claim("method", shared_text)
    app_a = _claim("applications", shared_text)

    result = assess_applications(
        [(paper_a, "Paper A", NEAR, [app_a]), (paper_b, "Paper B", NEAR, [task_b])], embedder
    )

    assert result.status == "found"
    assert len(result.applications) == 1
```

Note: `test_excludes_stub_claims` from the old file is deliberately removed, not overlooked — stub-filtering is no longer this module's responsibility. `assess_applications` no longer queries the database; it trusts the caller (`build_assessment`, via the existing `_claims_for_paper` helper) to have already excluded stub claims, exactly as every other `assess_*` function consuming the same `papers_with_claims` bundle already does.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_applications.py -v`
Expected: FAIL — `assess_applications` doesn't accept this signature yet (`TypeError`).

- [ ] **Step 3: Rewrite `assessment/applications.py`**

```python
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

# Provisional - a reasoned starting point (higher than gaps/signals.py's
# 0.85, since this check's job has narrowed to catching only near-total
# paraphrase, with Gate 1's lexical check doing the primary
# discrimination), NOT yet verified against real data. See the live-
# corpus spot-check task in docs/superpowers/plans/2026-09-02-
# applications-evidence-grounding.md before trusting this number.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_applications.py -v`
Expected: PASS — all 13 tests.

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/applications.py tests/test_assessment_applications.py
git commit -m "feat(assessment): add own-task-overlap check, make assess_applications a pure function"
```

---

## Task 3: Update `build.py`'s call site

**Files:**
- Modify: `src/researchbridge/assessment/build.py`
- Test: `tests/test_assessment_build.py` (no new tests expected — this is a regression-verification task)

**Interfaces:**
- Consumes: `assess_applications(papers_with_claims: list[PaperWithClaims], embedder: Embedder)` from Task 2.

- [ ] **Step 1: Update the call site**

In `src/researchbridge/assessment/build.py`, find:

```python
    applications = assess_applications(session, papers_by_distance)
```

Replace with:

```python
    applications = assess_applications(
        [(paper.id, paper.title, distance, claims) for paper, distance, claims in papers_with_claims], embedder
    )
```

This line must come after `papers_with_claims` is assembled (already the case — it's assembled once near the top of `build_assessment`, before any `assess_*` call) and can sit anywhere among the other `assess_*` calls; keep it next to `gap = assess_research_gap(...)` where it already is, for readability.

- [ ] **Step 2: Run the full assessment test suite to verify no regressions**

Run: `pytest tests/ -k "assessment" -v`
Expected: PASS across every assessment-related test file, including all pre-existing `test_assessment_build.py` tests that create `"applications"`-type claims directly via the test file's own `_claim` helper (e.g. `test_potential_applications_is_set_from_the_neighborhood`, `test_applications_evidence_is_linked_with_role_application`) — these bypass Gate 1 entirely (they insert `ExtractedClaim` rows directly, not through the extraction pipeline) and are not expected to trip Gate 2 either, since none of their fixtures pair an `"applications"` claim with a same-paper `problem`/`method`/`main_contribution`/`results` claim close enough (under the test file's hash-based `FakeEmbedder`) to cross `OWN_TASK_OVERLAP_THRESHOLD`. If any of them unexpectedly fail, that is a real signal worth investigating before proceeding, not something to work around by loosening the threshold ad hoc.

- [ ] **Step 3: Commit**

```bash
git add src/researchbridge/assessment/build.py
git commit -m "feat(assessment): wire the pure-function assess_applications into build_assessment"
```

---

## Task 4: Live-corpus calibration spot-check for `OWN_TASK_OVERLAP_THRESHOLD`

This task is deliberately not a pytest file — same manual-calibration pattern already used for `DIMENSION_MATCH_SIMILARITY` (`coverage.py`) and every other real-embedder threshold in this codebase (`novelty.py`'s `NEAR_DISTANCE`/`FAR_DISTANCE`, `gaps/signals.py`'s `OWN_CONTRIBUTION_OVERLAP_THRESHOLD`).

- [ ] **Step 1: Find or construct 3-5 real same-paper (task claim, application claim) pairs from the live corpus**

Using the real `SentenceTransformerEmbedder` against papers already in the dev database (or a small scratch script following the same pattern as prior calibration passes in this project — read the corpus, find papers with both a `method`/`main_contribution`/`results` claim and an `applications` claim, compute their cosine similarity), gather real pairs spanning: (a) a genuine external application with real actor/action language (expected: overlap comfortably below 0.92), (b) if any exist in the corpus, a borderline case where the application claim closely echoes the task (expected: overlap approaching or exceeding 0.92).

- [ ] **Step 2: Inspect the real similarity numbers and adjust `OWN_TASK_OVERLAP_THRESHOLD` if needed**

If genuine applications in the real corpus sit closer to 0.92 than expected (risking false rejection), or if no real near-paraphrase pair ever approaches 0.92 (suggesting the threshold is too high to ever fire), adjust the constant in `src/researchbridge/assessment/applications.py` accordingly.

- [ ] **Step 3: Document the calibration**

Update `OWN_TASK_OVERLAP_THRESHOLD`'s docstring comment in `applications.py` with the same kind of calibration note `novelty.py`/`gaps/signals.py` already carry: what was checked, what the raw numbers were, why the final value was picked. Re-run `pytest tests/test_assessment_applications.py -v` afterward to confirm nothing broke while editing the file.

- [ ] **Step 4: Commit the documentation update**

```bash
git add src/researchbridge/assessment/applications.py
git commit -m "docs(assessment): document live-corpus calibration for OWN_TASK_OVERLAP_THRESHOLD"
```

---

## Task 5: Re-run the extraction benchmark and record the before/after comparison

**Files:**
- Modify: `benchmark/extraction_eval_results.json` (regenerated, not hand-edited)

- [ ] **Step 1: Re-run the benchmark evaluator**

```bash
rb-extract-evaluate
```

This overwrites `benchmark/extraction_eval_results.json` and prints both tables to stdout: per-extractor field precision/recall/F1 (`evaluate()`), and the claim-type-validation table (`evaluate_claim_type_validation()` — own-field-acceptance-rate and cross-field-rejection-rate per validated field).

- [ ] **Step 2: Compare against the pre-change baseline**

Baseline (recorded in the design spec, generated 2026-08-29, before this change): hybrid extractor `applications` precision 0.417, recall 0.303. Confirm:
- Precision moves upward from this baseline (the primary goal of this change).
- `applications`' own-field-acceptance-rate in the type-validation table does not collapse — a large drop would mean the stricter regex is now rejecting genuinely-correct ground-truth application text (a false-negative regression), which would mean Task 1's regex needs revisiting, not that this step should be skipped.
- `applications`' cross-field-rejection-rate does not regress from wherever it stood before.

If precision does not improve, or own-field-acceptance-rate drops sharply, stop and investigate before proceeding — do not commit a benchmark result that doesn't support the change.

- [ ] **Step 3: Commit the updated benchmark result**

```bash
git add benchmark/extraction_eval_results.json
git commit -m "chore(benchmark): re-run extraction eval after applications validation rewrite"
```

---

## Self-review notes (for the plan author, not a task to execute)

- **Spec coverage:** Gate 1 (Task 1) and Gate 2 (Task 2) implement the design spec's two-gate architecture exactly as resolved in the own-task-overlap discussion; Task 3 is the necessary wiring the spec's "Necessary changes, by layer" table calls for; Task 4 is the spec's explicitly-flagged threshold-calibration follow-up; Task 5 is the spec's benchmark-evaluation requirement. The spec's "Out of scope" list (API/frontend exposure, idea-specific relevance, `tier="weak"`, shared `ClaimRole` framework, `heuristic.py` changes, `coverage.py` changes, cross-paper synthesis, new calibration infrastructure) has no corresponding task, correctly.
- **Placeholder scan:** every step has literal, complete code or a literal command — no "TBD"/"handle appropriately" patterns.
- **Type consistency:** `ClaimRecord` and `PaperWithClaims` are defined once, in Task 2, and Task 3's `build.py` edit constructs exactly that shape from the `papers_with_claims` tuple `build_assessment` already has (`paper.id, paper.title, distance, claims`) — matches Task 2's `assess_applications(papers_with_claims: list[PaperWithClaims], embedder: Embedder)` signature exactly. `ApplicationRecord`/`ApplicationsResult` are unchanged in shape from today (no `tier` field added, per the spec's explicit decision) — Task 1's `ValidationResult.tier="strong"` is persisted via the existing extraction pipeline machinery (`extraction/pipeline.py`, untouched by this plan) and never threaded into `ApplicationRecord`, consistent with "no API/frontend changes in this plan."
