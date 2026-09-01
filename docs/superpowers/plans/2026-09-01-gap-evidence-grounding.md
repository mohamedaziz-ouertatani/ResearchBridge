# Gap Evidence Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop candidate gaps from being generated out of recurring *topics* or *paper motivation* by requiring deterministic, corpus-grounded evidence of unresolvedness before a cluster of limitation/research_gap claims is allowed to become a reviewable `CandidateGap`.

**Architecture:** Extend the existing extractive, non-LLM pipeline with a new pure-function classification layer (`gaps/signals.py`) that scores each `limitations`/`research_gap` claim against two negative signals (same-sentence self-resolution language, and semantic overlap with the *same paper's own* `main_contribution`/`results` claims) to assign a role (`anchor` / `supporting` / `motivation`), then aggregates cluster membership into one of `strong_gap` / `potential_gap` / `known_limitation`, or drops the cluster entirely (`insufficient_evidence`) before it is ever persisted. A separate cross-paper "addressing signal" check downgrades (never invalidates) a cluster when another paper in the corpus already appears to address it. Orchestration in `gaps/detect.py` also distinguishes, per seed paper, "no relevant papers retrieved" from "relevant papers but no evidence cleared the bar" from "gaps found" — mirroring the existing `assessment/coverage.py::DimensionCoverage` precedent for that exact distinction.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.0 ORM, Alembic migrations, pytest, scikit-learn (`AgglomerativeClustering`, unchanged), Next.js/TypeScript frontend (App Router), no new dependencies.

**Spec:** This plan implements the design agreed in conversation (no separate spec file exists — the "Revised design" and its follow-up adjustments in this session are the spec). Key decisions carried into every task below:
- Precision over recall: fewer, defensible candidates beats more plausible-sounding ones.
- `self_resolution` and `own_contribution_overlap` are each independent, partial negative evidence — never individually sufficient to declare a gap resolved.
- No LLM, anywhere. Every new signal is regex/lexicon or cosine-similarity based, exactly like the existing `extraction/validation.py` and `gaps/cluster.py`.
- `DimensionCoverage` (`assessment/coverage.py`) is explicitly out of scope and must not be touched or given a DB table — it already is the in-memory, non-persisted pattern this plan follows for gap-detection status.

## Global Constraints

- No LLM calls anywhere in this pipeline — every new function must be regex/lexicon or embedding-cosine-similarity based, deterministic, and unit-testable without network access.
- Every new DB column is nullable (existing rows predate this work and must not break); no backfill migration is required.
- `gaps/cluster.py` (`ClaimRecord`, `GapCluster`, `find_recurring_patterns`) stays untouched — its existing 10-test suite (`tests/test_gaps_cluster.py`) must keep passing unmodified. All new classification logic lives in the new `gaps/signals.py` module and is wired in by `gaps/detect.py` around the existing clustering call.
- Follow existing code style exactly: module docstrings explain *why*, not what; constants get a comment citing how they were calibrated (or a TODO citing the calibration task below if not yet calibrated against the real embedder); dataclasses over dicts for structured returns; batched `embed_texts()` calls, never one string at a time in a loop.
- Do not create or modify anything under `assessment/coverage.py` or any `assessment/*` file — that subsystem is a precedent to imitate, not a dependency to change.

---

## File Structure

| File | Responsibility |
|---|---|
| `migrations/versions/0014_extraction_validation_tier.py` | New | Adds `extracted_claims.validation_tier` |
| `migrations/versions/0015_gap_status_and_provenance.py` | New | Adds `candidate_gaps.gap_status`/`.resolution_note` and `candidate_gap_evidence.claim_role`/`.self_resolution_signal`/`.field_scope_signal`/`.own_contribution_overlap` |
| `src/researchbridge/db/models.py` | Modify | Add the 6 new columns above to `ExtractedClaim`, `CandidateGap`, `CandidateGapEvidence` |
| `src/researchbridge/extraction/validation.py` | Modify | `ValidationResult` gains a `tier` field; `validate_claim_type` sets it for `research_gap` |
| `src/researchbridge/extraction/pipeline.py` | Modify | Persist `type_validation.tier` onto the new `ExtractedClaim.validation_tier` column |
| `src/researchbridge/gaps/signals.py` | New | Pure classification: `classify_claim`, `classify_cluster`, `find_addressing_papers`, `apply_addressing_downgrade`, the two new regexes, `cosine_similarity` |
| `src/researchbridge/gaps/detect.py` | Modify | Load contribution claims, compute overlap, classify claims/clusters, apply addressing downgrade, return `GapDetectionResult` (was: bare `list[CandidateGapDraft]`) |
| `src/researchbridge/gaps/persistence.py` | Modify | Persist `gap_status`/`resolution_note` on `CandidateGap` and per-evidence classification on `CandidateGapEvidence` |
| `src/researchbridge/gaps/batch.py` | Modify | Consume `GapDetectionResult`; track `no_relevant_papers`/`insufficient_evidence` counts in `BatchSummary` |
| `src/researchbridge/gaps/cli_detect.py` | Modify | Print the per-seed detection status; consume `GapDetectionResult` |
| `src/researchbridge/api/schemas.py` | Modify | `CandidateGapOut` gains `gap_status`/`resolution_note`; `GapEvidenceOut` gains `claim_type`/`validation_tier`/`claim_role`/`self_resolution_signal`/`field_scope_signal`/`own_contribution_overlap` |
| `src/researchbridge/api/serializers.py` | Modify | `_evidence_by_gap` joins `ExtractedClaim` for `validation_tier`/`claim_type` and reads the new `CandidateGapEvidence` columns |
| `frontend/lib/gapsApi.ts` | Modify | Mirror the new API fields in `CandidateGap`/`GapEvidence` types |
| `frontend/app/gaps/page.tsx` | Modify | Render a `gap_status` badge, per-evidence `claim_type`/`claim_role` badges, and the `resolution_note` callout |
| `tests/test_extraction_validation.py` | Modify | Add tier assertions to existing tests |
| `tests/test_extraction_pipeline.py` | Modify | Assert `validation_tier` is persisted |
| `tests/test_gaps_signals.py` | New | Unit tests for `classify_claim`/`classify_cluster`/`find_addressing_papers`/`apply_addressing_downgrade` |
| `tests/test_gaps_detect.py` | Modify | Update for `GapDetectionResult`; add overlap/self-resolution/status coverage |
| `tests/test_gaps_persistence.py` | Modify | Assert new columns are written |
| `tests/test_gaps_batch.py` | Modify | Update for `GapDetectionResult`; assert new summary counters |
| `tests/test_gaps_cli_detect.py` | Modify | Update for `GapDetectionResult` |
| `tests/test_gaps_api.py` | Modify | Assert new response fields |
| `tests/test_gaps_golden_regression.py` | New | The 7-scenario golden set from the design discussion, end-to-end through `detect_candidate_gaps` |
| `src/researchbridge/gaps/calibration.py` | New | Pure evaluation functions: builds threshold-calibration samples from the Sec 25 hand-annotated benchmark and curated cross-paper groups, sweeps candidate thresholds, scores false-positive/false-negative rates per category |
| `src/researchbridge/gaps/cli_calibrate.py` | New | CLI: runs `calibration.py` against the real corpus + real embedder, prints a per-category/per-threshold table, writes `benchmark/gap_calibration_results.json` (mirrors `extraction/cli_evaluate.py`) |
| `benchmark/gap_calibration_groups.yaml` | New | Hand-curated cross-paper groups (recurring genuine limitation, recurring topical convergence, benchmark-overlap-not-solving) that the per-paper Sec 25 annotations can't represent alone |
| `tests/test_gaps_calibration.py` | New | Unit tests for `calibration.py`'s sampling/sweep/scoring logic against fake annotations and a fake embedder |

---

## Task 1: Persist the claim-validation tier

**Files:**
- Modify: `src/researchbridge/extraction/validation.py`
- Modify: `src/researchbridge/extraction/pipeline.py:183-191`
- Modify: `src/researchbridge/db/models.py:193-210` (`ExtractedClaim`)
- Create: `migrations/versions/0014_extraction_validation_tier.py`
- Test: `tests/test_extraction_validation.py`
- Test: `tests/test_extraction_pipeline.py`

**Interfaces:**
- Produces: `ValidationResult.tier: str | None` (`"strong"` | `"weak"` | `None`) — consumed by Task 6.
- Produces: `ExtractedClaim.validation_tier: str | None` DB column — consumed by Task 6's `detect.py` query.

Right now `validate_claim_type` returns pass/fail and discards *which* regex tier matched. Persisting that tier is what lets clustering later require an unambiguous ("strong") anchor claim rather than treating "future research" boilerplate the same as "remains an open problem."

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_extraction_validation.py`:

```python
def test_strong_gap_language_sets_strong_tier() -> None:
    text = "Extending this approach to multilingual settings remains an open problem for future work."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "strong"


def test_weak_gap_language_sets_weak_tier() -> None:
    text = "This finding suggests a promising direction for future research in the area."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is True
    assert result.tier == "weak"


def test_non_research_gap_types_have_no_tier() -> None:
    text = "However, the approach does not scale to graphs with more than a million nodes."
    result = validate_claim_type("limitations", text)

    assert result.is_valid is True
    assert result.tier is None


def test_rejected_claim_has_no_tier() -> None:
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    result = validate_claim_type("research_gap", text)

    assert result.is_valid is False
    assert result.tier is None
```

Add to `tests/test_extraction_pipeline.py` (find the existing test that asserts an `ExtractedClaim` row is created for a valid `research_gap` candidate — extend its assertions):

```python
def test_research_gap_claim_persists_its_validation_tier(session_factory) -> None:
    from researchbridge.db.models import ExtractedClaim, Paper
    from researchbridge.extraction.base import ClaimCandidate
    from researchbridge.extraction.pipeline import ExtractionPipeline

    session = session_factory()
    paper = Paper(
        source="fake", source_id="tier-1",
        title="t", abstract="Extending this approach to multilingual settings remains an open problem for future work.",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()

    class _Extractor:
        extraction_method = "fake"
        model_version = "fake-v1"

        def extract(self, paper):
            return [
                ClaimCandidate(
                    claim_type="research_gap",
                    claim_text=paper.abstract,
                    evidence_quote=paper.abstract,
                    confidence="medium",
                )
            ]

    pipeline = ExtractionPipeline(_Extractor(), session_factory)
    pipeline.run()

    claim = session.execute(select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)).scalar_one()
    assert claim.validation_tier == "strong"
    session.close()
```

Add `from sqlalchemy import select` to the top of `tests/test_extraction_pipeline.py` if not already imported — check first.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction_validation.py tests/test_extraction_pipeline.py -v -k "tier"`
Expected: FAIL — `ValidationResult` has no `tier` attribute; `ExtractedClaim` has no `validation_tier` attribute.

- [ ] **Step 3: Add `tier` to `ValidationResult` and set it**

In `src/researchbridge/extraction/validation.py`, change the dataclass and the `research_gap` branch:

```python
@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str | None = None
    tier: str | None = None
    """"strong" or "weak" for an accepted research_gap claim (which regex
    tier matched) - None for every other claim_type, and None for any
    rejected claim. Persisted downstream (extracted_claims.validation_tier)
    so gap clustering can require an unambiguous anchor claim rather than
    treating "future research" boilerplate the same as "remains unresolved"."""
```

```python
    if claim_type == "research_gap":
        if _STRONG_GAP_LANGUAGE_RE.search(text):
            return ValidationResult(True, tier="strong")
        if _WEAK_GAP_LANGUAGE_RE.search(text) and not _has_result_signal(text):
            return ValidationResult(True, tier="weak")
        return ValidationResult(
            False, "no unresolved-problem/future-work language found; not a stated research gap"
        )
```

- [ ] **Step 4: Add the DB column**

In `src/researchbridge/db/models.py`, inside `class ExtractedClaim`, after the `confidence` column (line 209):

```python
    validation_tier: Mapped[str | None] = mapped_column(String, nullable=True)
    """"strong"/"weak" for a research_gap claim that passed extraction
    validation (see extraction/validation.py::ValidationResult.tier) - None
    for every other claim_type. Read by gaps/detect.py to decide whether a
    claim can anchor a candidate gap on its own."""
```

- [ ] **Step 5: Write the migration**

Create `migrations/versions/0014_extraction_validation_tier.py`:

```python
"""extracted_claims: add validation_tier

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("extracted_claims", sa.Column("validation_tier", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("extracted_claims", "validation_tier")
```

Check `migrations/versions/0013_citation_fetch_runs.py`'s `revision`/`down_revision` values before writing this — confirm `0013` is really the current head (`alembic heads`) rather than assuming.

- [ ] **Step 6: Persist the tier in the pipeline**

In `src/researchbridge/extraction/pipeline.py:183-191`, add `validation_tier=type_validation.tier` to the `ExtractedClaim(...)` constructor call:

```python
                    session.add(
                        ExtractedClaim(
                            paper_id=paper.id,
                            claim_type=candidate.claim_type,
                            text=candidate.claim_text,
                            evidence_id=evidence.id,
                            confidence=candidate.confidence,
                            validation_tier=type_validation.tier,
                        )
                    )
```

- [ ] **Step 7: Run the migration and tests**

Run: `alembic upgrade head`
Run: `pytest tests/test_extraction_validation.py tests/test_extraction_pipeline.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/researchbridge/extraction/validation.py src/researchbridge/extraction/pipeline.py src/researchbridge/db/models.py migrations/versions/0014_extraction_validation_tier.py tests/test_extraction_validation.py tests/test_extraction_pipeline.py
git commit -m "feat(extraction): persist which regex tier validated a research_gap claim"
```

---

## Task 2: `gaps/signals.py` — per-claim classification (`classify_claim`)

**Files:**
- Create: `src/researchbridge/gaps/signals.py`
- Test: `tests/test_gaps_signals.py`

**Interfaces:**
- Produces: `ClaimRole = Literal["anchor", "supporting", "motivation"]`
- Produces: `ClaimClassification` dataclass: `role: ClaimRole`, `self_resolution: bool`, `field_scope: bool`, `own_contribution_overlap: float`
- Produces: `classify_claim(text: str, claim_type: str, gap_tier: str | None, own_contribution_overlap: float) -> ClaimClassification`
- Produces: `OWN_CONTRIBUTION_OVERLAP_THRESHOLD: float`, `cosine_similarity(a: list[float], b: list[float]) -> float`
- Consumed by: Task 3 (`classify_cluster`), Task 6 (`detect.py`)

This is the core deterministic-classification function discussed in the design: `self_resolution` and `own_contribution_overlap` are each a *partial* negative signal (never alone conclusive), `limitations` claims are always treated as tier `"weak"` (they have no strong/weak split of their own — only an explicit `research_gap` claim can single-handedly earn `"anchor"`), and a claim needs *both* self-resolution language *and* high contribution overlap to be excluded outright as `"motivation"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gaps_signals.py`:

```python
from __future__ import annotations

from researchbridge.gaps.signals import (
    OWN_CONTRIBUTION_OVERLAP_THRESHOLD,
    ClaimClassification,
    classify_claim,
    cosine_similarity,
)


def test_strong_gap_with_field_scope_and_no_negative_signal_is_anchor() -> None:
    text = "Despite recent progress, robustness to adversarial perturbations remains unresolved for existing methods."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.role == "anchor"
    assert result.self_resolution is False
    assert result.field_scope is True


def test_strong_gap_without_field_scope_language_is_only_supporting() -> None:
    # unambiguous unresolved-language, but no explicit third-person/field framing -
    # still real signal, just not enough alone to anchor a strong_gap
    text = "This remains an open problem for future work."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_self_resolution_language_downgrades_a_strong_anchor_to_supporting() -> None:
    text = "This paper investigates whether large language models can help bridge this gap."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.self_resolution is True
    assert result.role != "anchor"


def test_high_own_contribution_overlap_alone_downgrades_but_does_not_exclude() -> None:
    # the CentaurBench-adjacent case for a paper that only PARTIALLY overlaps -
    # e.g. a benchmark/evaluation paper: overlap is real negative evidence,
    # but on its own (no self-resolution language) it must not fully exclude
    # the claim, per the "benchmarking is not solving" requirement
    text = "No existing benchmark evaluates robustness under realistic table perturbations."
    result = classify_claim(
        text, "research_gap", gap_tier="strong",
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.self_resolution is False
    assert result.role == "supporting"  # demoted from anchor, not excluded


def test_self_resolution_and_high_overlap_together_exclude_as_motivation() -> None:
    # both signals firing together is the "prior work has X; we address X"
    # pattern the design calls out explicitly - this is the one case that
    # should fully exclude the claim from clustering
    text = "This paper investigates whether large language models can help bridge this gap."
    result = classify_claim(
        text, "research_gap", gap_tier="strong",
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.role == "motivation"


def test_weak_gap_tier_can_only_ever_be_supporting_or_motivation_never_anchor() -> None:
    text = "This finding suggests a promising direction for future research in the area."
    result = classify_claim(text, "research_gap", gap_tier="weak", own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_limitations_claim_type_is_always_treated_as_weak_tier() -> None:
    # limitations has no strong/weak split of its own (validation.py never
    # sets a tier for it) - by design it can corroborate but never single-
    # handedly anchor a strong_gap
    text = "However, the approach does not scale to graphs with more than a million nodes despite existing methods."
    result = classify_claim(text, "limitations", gap_tier=None, own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_limitations_claim_with_self_resolution_and_overlap_is_motivation() -> None:
    text = "We propose a graph-based method to address this limitation."
    result = classify_claim(
        text, "limitations", gap_tier=None,
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.role == "motivation"


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.gaps.signals'`

- [ ] **Step 3: Write `gaps/signals.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_signals.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/signals.py tests/test_gaps_signals.py
git commit -m "feat(gaps): add deterministic per-claim gap-vs-motivation classification"
```

---

## Task 3: `gaps/signals.py` — cluster-level status (`classify_cluster`)

**Files:**
- Modify: `src/researchbridge/gaps/signals.py`
- Test: `tests/test_gaps_signals.py`

**Interfaces:**
- Consumes: `ClaimClassification` from Task 2
- Produces: `ClassifiedMember` dataclass: `paper_id: uuid.UUID`, `evidence_id: uuid.UUID`, `text: str`, `classification: ClaimClassification`
- Produces: `classify_cluster(members: list[ClassifiedMember], min_cluster_size: int) -> GapStatus | None`
- Consumed by: Task 6 (`detect.py`)

The decision table: `strong_gap` needs 2+ distinct papers whose claim classified as `anchor`. `potential_gap` needs exactly 1 anchor paper plus enough *independently corroborating* supporting papers (papers whose claim is NOT itself overlap-flagged — i.e. not explainable by that same paper's own contribution) to clear `min_cluster_size`. `known_limitation` needs `min_cluster_size` independently-corroborating papers with no anchor at all. Everything else returns `None` (drop, never persisted). The "independent" requirement is what keeps a cluster of overlap-flagged benchmark-motivation claims (the FinRCA-Bench pattern) from being accepted merely because 3 papers happen to say something similar — every one of them being explainable by that paper's own contribution means there's no cross-paper corroboration of anything *unresolved*, only of a shared research topic.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gaps_signals.py`:

```python
import uuid

from researchbridge.gaps.signals import ClassifiedMember, classify_cluster


def _member(paper_id: uuid.UUID | None, role: str, overlap: float = 0.0) -> ClassifiedMember:
    return ClassifiedMember(
        paper_id=paper_id or uuid.uuid4(),
        evidence_id=uuid.uuid4(),
        text="claim text",
        classification=ClaimClassification(
            role=role, self_resolution=False, field_scope=(role == "anchor"), own_contribution_overlap=overlap
        ),
    )


def test_two_anchor_papers_is_strong_gap() -> None:
    members = [_member(None, "anchor"), _member(None, "anchor"), _member(None, "supporting")]

    assert classify_cluster(members, min_cluster_size=3) == "strong_gap"


def test_one_anchor_with_enough_independent_support_is_potential_gap() -> None:
    members = [
        _member(None, "anchor"),
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) == "potential_gap"


def test_one_anchor_with_only_overlap_flagged_support_is_insufficient() -> None:
    # the anchor paper alone can't clear min_cluster_size, and the
    # "corroborating" papers are all explainable by their own contribution -
    # not independent corroboration of anything unresolved
    members = [
        _member(None, "anchor"),
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_three_independent_supporting_papers_with_no_anchor_is_known_limitation() -> None:
    members = [
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) == "known_limitation"


def test_all_overlap_flagged_supporting_papers_is_insufficient_evidence() -> None:
    # the FinRCA-Bench pattern: three benchmark papers each independently
    # motivating their own contribution with near-identical "existing
    # benchmarks lack X" phrasing, each demoted for overlapping with their
    # OWN contribution - recurring topic, not corroborated unresolvedness
    members = [
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_only_motivation_members_is_insufficient_evidence() -> None:
    members = [_member(None, "motivation"), _member(None, "motivation"), _member(None, "motivation")]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_same_paper_repeating_does_not_count_twice_toward_the_minimum() -> None:
    same_paper = uuid.uuid4()
    members = [
        _member(same_paper, "supporting", overlap=0.1),
        _member(same_paper, "supporting", overlap=0.1),
        _member(same_paper, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_signals.py -v -k cluster`
Expected: FAIL — `classify_cluster`/`ClassifiedMember` not defined.

- [ ] **Step 3: Implement**

Append to `src/researchbridge/gaps/signals.py`:

```python
import uuid


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_signals.py -v`
Expected: PASS (all tests in the file, including Task 2's)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/signals.py tests/test_gaps_signals.py
git commit -m "feat(gaps): aggregate classified claims into a cluster gap status"
```

---

## Task 4: `gaps/signals.py` — cross-paper addressing signal

**Files:**
- Modify: `src/researchbridge/gaps/signals.py`
- Test: `tests/test_gaps_signals.py`

**Interfaces:**
- Produces: `AddressingMatch` dataclass: `paper_id: uuid.UUID`, `paper_title: str`, `text: str`, `similarity: float`
- Produces: `find_addressing_papers(representative_vector: list[float], candidates: list[tuple[uuid.UUID, str, str, list[float]]], threshold: float = ADDRESSING_SIMILARITY_THRESHOLD) -> list[AddressingMatch]`
- Produces: `apply_addressing_downgrade(status: GapStatus, matches: list[AddressingMatch]) -> tuple[GapStatus, str | None]`
- Consumed by: Task 6 (`detect.py`)

Renamed per the design adjustment from "resolution" to "addressing" throughout: finding that some paper's contribution/results claim is similar to the gap is evidence worth a note and a one-level downgrade, never an automatic invalidation (a paper can benchmark, partially address, or provide infrastructure around a problem without solving it).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gaps_signals.py`:

```python
from researchbridge.gaps.signals import AddressingMatch, apply_addressing_downgrade, find_addressing_papers


def test_finds_a_paper_whose_contribution_is_similar_enough() -> None:
    seed_id = uuid.uuid4()
    other_id = uuid.uuid4()
    candidates = [
        (seed_id, "Seed Paper", "an unrelated contribution", [0.0, 1.0]),
        (other_id, "Other Paper", "a benchmark for robustness to table perturbations", [1.0, 0.0]),
    ]

    matches = find_addressing_papers([1.0, 0.0], candidates, threshold=0.5)

    assert len(matches) == 1
    assert matches[0].paper_id == other_id
    assert matches[0].similarity == 1.0


def test_no_match_below_threshold() -> None:
    candidates = [(uuid.uuid4(), "Other Paper", "text", [0.0, 1.0])]

    matches = find_addressing_papers([1.0, 0.0], candidates, threshold=0.5)

    assert matches == []


def test_strong_gap_downgrades_to_potential_gap_when_addressed() -> None:
    match = AddressingMatch(paper_id=uuid.uuid4(), paper_title="Other Paper", text="text", similarity=0.7)

    status, note = apply_addressing_downgrade("strong_gap", [match])

    assert status == "potential_gap"
    assert note is not None
    assert "Other Paper" in note


def test_potential_gap_downgrades_to_known_limitation_when_addressed() -> None:
    match = AddressingMatch(paper_id=uuid.uuid4(), paper_title="Other Paper", text="text", similarity=0.7)

    status, note = apply_addressing_downgrade("potential_gap", [match])

    assert status == "known_limitation"


def test_known_limitation_stays_known_limitation_but_still_gets_a_note() -> None:
    match = AddressingMatch(paper_id=uuid.uuid4(), paper_title="Other Paper", text="text", similarity=0.7)

    status, note = apply_addressing_downgrade("known_limitation", [match])

    assert status == "known_limitation"
    assert note is not None


def test_no_matches_leaves_status_and_note_untouched() -> None:
    status, note = apply_addressing_downgrade("strong_gap", [])

    assert status == "strong_gap"
    assert note is None


def test_downgrade_uses_the_highest_similarity_match() -> None:
    weak = AddressingMatch(paper_id=uuid.uuid4(), paper_title="Weak Match", text="t", similarity=0.51)
    strong = AddressingMatch(paper_id=uuid.uuid4(), paper_title="Strong Match", text="t", similarity=0.9)

    _, note = apply_addressing_downgrade("strong_gap", [weak, strong])

    assert "Strong Match" in note
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_signals.py -v -k addressing`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement**

Append to `src/researchbridge/gaps/signals.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_signals.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/signals.py tests/test_gaps_signals.py
git commit -m "feat(gaps): add cross-paper addressing-signal downgrade"
```

---

## Task 5: DB migration + model columns for gap status and evidence provenance

**Files:**
- Modify: `src/researchbridge/db/models.py:279-350` (`CandidateGap`, `CandidateGapEvidence`)
- Create: `migrations/versions/0015_gap_status_and_provenance.py`

**Interfaces:**
- Produces: `CandidateGap.gap_status: str | None`, `CandidateGap.resolution_note: str | None`
- Produces: `CandidateGapEvidence.claim_role: str | None`, `.self_resolution_signal: bool | None`, `.field_scope_signal: bool | None`, `.own_contribution_overlap: float | None`
- Consumed by: Task 7 (`persistence.py`), Task 11 (serializers/schemas)

- [ ] **Step 1: Add the columns to the models**

In `src/researchbridge/db/models.py`, `class CandidateGap`, add after `usefulness_rating` (line 325):

```python
    gap_status: Mapped[str | None] = mapped_column(String, nullable=True)
    """"strong_gap"/"potential_gap"/"known_limitation" - the deterministic
    classification from gaps/signals.py::classify_cluster, computed once at
    detection time and stored so the review UI and API don't recompute it.
    NULL for any CandidateGap saved before this column existed."""
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Set only when gaps/signals.py::find_addressing_papers found another
    paper in the corpus whose contribution/results claim closely resembles
    this pattern - a downgrade note for the reviewer, never proof the gap
    is closed (see apply_addressing_downgrade)."""
```

Also add the CHECK constraint to `__table_args__` (line 306-308):

```python
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_candidate_gaps_status"),
        CheckConstraint(
            "gap_status IS NULL OR gap_status IN ('strong_gap', 'potential_gap', 'known_limitation')",
            name="ck_candidate_gaps_gap_status",
        ),
    )
```

In `class CandidateGapEvidence`, add after `evidence_id` (line 349):

```python
    claim_role: Mapped[str | None] = mapped_column(String, nullable=True)
    """"anchor"/"supporting" - this evidence row's classify_claim role
    within its candidate gap's cluster (gaps/signals.py). NULL for rows
    saved before this column existed."""
    self_resolution_signal: Mapped[bool | None] = mapped_column(nullable=True)
    field_scope_signal: Mapped[bool | None] = mapped_column(nullable=True)
    own_contribution_overlap: Mapped[float | None] = mapped_column(Float, nullable=True)
```

Also add a CHECK constraint to `__table_args__` (line 341-343):

```python
    __table_args__ = (
        UniqueConstraint("candidate_gap_id", "evidence_id", name="uq_candidate_gap_evidence_gap_evidence"),
        CheckConstraint(
            "claim_role IS NULL OR claim_role IN ('anchor', 'supporting')",
            name="ck_candidate_gap_evidence_claim_role",
        ),
    )
```

Check the top of `models.py` for the existing `Float` import (used elsewhere, e.g. `CandidateGap.similarity_threshold`) — it should already be imported; if not, add it to the `sqlalchemy` import line.

- [ ] **Step 2: Write the migration**

Create `migrations/versions/0015_gap_status_and_provenance.py`:

```python
"""candidate_gaps/candidate_gap_evidence: gap_status, resolution_note, and
per-evidence classification provenance

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

GAP_STATUS_CONSTRAINT = "ck_candidate_gaps_gap_status"
CLAIM_ROLE_CONSTRAINT = "ck_candidate_gap_evidence_claim_role"


def upgrade() -> None:
    op.add_column("candidate_gaps", sa.Column("gap_status", sa.String(), nullable=True))
    op.add_column("candidate_gaps", sa.Column("resolution_note", sa.Text(), nullable=True))
    op.create_check_constraint(
        GAP_STATUS_CONSTRAINT,
        "candidate_gaps",
        "gap_status IS NULL OR gap_status IN ('strong_gap', 'potential_gap', 'known_limitation')",
    )

    op.add_column("candidate_gap_evidence", sa.Column("claim_role", sa.String(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("self_resolution_signal", sa.Boolean(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("field_scope_signal", sa.Boolean(), nullable=True))
    op.add_column("candidate_gap_evidence", sa.Column("own_contribution_overlap", sa.Float(), nullable=True))
    op.create_check_constraint(
        CLAIM_ROLE_CONSTRAINT,
        "candidate_gap_evidence",
        "claim_role IS NULL OR claim_role IN ('anchor', 'supporting')",
    )


def downgrade() -> None:
    op.drop_constraint(CLAIM_ROLE_CONSTRAINT, "candidate_gap_evidence", type_="check")
    op.drop_column("candidate_gap_evidence", "own_contribution_overlap")
    op.drop_column("candidate_gap_evidence", "field_scope_signal")
    op.drop_column("candidate_gap_evidence", "self_resolution_signal")
    op.drop_column("candidate_gap_evidence", "claim_role")

    op.drop_constraint(GAP_STATUS_CONSTRAINT, "candidate_gaps", type_="check")
    op.drop_column("candidate_gaps", "resolution_note")
    op.drop_column("candidate_gaps", "gap_status")
```

- [ ] **Step 3: Run the migration**

Run: `alembic upgrade head`
Expected: succeeds, no errors.

- [ ] **Step 4: Verify the models import and the app boots**

Run: `python -c "from researchbridge.db import models"`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/db/models.py migrations/versions/0015_gap_status_and_provenance.py
git commit -m "feat(db): add gap_status/resolution_note and per-evidence classification columns"
```

---

## Task 6: `detect.py` — compute own-contribution overlap and classify claims

**Files:**
- Modify: `src/researchbridge/gaps/detect.py`
- Test: `tests/test_gaps_detect.py`

**Interfaces:**
- Consumes: `classify_claim`, `ClaimClassification` from Task 2; `ExtractedClaim.validation_tier` from Task 1
- Produces (new, private to `detect.py`): `_GapClaimRow` dataclass (`paper_id`, `evidence_id`, `text`, `claim_type`, `gap_tier`); `_load_gap_claims`, `_load_contribution_claims`, `_own_contribution_overlaps`
- Does not change `gaps/cluster.py`'s public interface at all — `find_recurring_patterns` keeps being called exactly as before.

This task only adds the overlap computation and per-claim classification *around* the untouched clustering call. Cluster-status aggregation and persistence land in Task 7-8; this task's tests check the classification inputs are computed correctly in isolation via a small `detect.py`-private helper exposed for testing.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gaps_detect.py` (reuse the file's existing `_paper`/`_claim`/`WordOverlapEmbedder` fixtures):

```python
from researchbridge.gaps.detect import _load_contribution_claims, _load_gap_claims, _own_contribution_overlaps


def test_load_gap_claims_reads_claim_type_and_tier(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "remains an open problem for future work")
    session.commit()

    rows = _load_gap_claims(session, [paper.id])

    session.close()
    assert len(rows) == 1
    assert rows[0].claim_type == "research_gap"
    assert rows[0].paper_id == paper.id


def test_load_contribution_claims_only_loads_contribution_and_results(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "main_contribution", "we build a new benchmark")
    _claim(session, paper, "results", "our method achieves 90% accuracy")
    _claim(session, paper, "method", "we use a transformer")
    session.commit()

    rows = _load_contribution_claims(session, [paper.id])

    session.close()
    assert {r.claim_type for r in rows} == {"main_contribution", "results"}


def test_own_contribution_overlap_is_high_for_near_identical_text(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "no existing benchmark evaluates robustness to table perturbations")
    _claim(session, paper, "main_contribution", "we introduce a benchmark evaluating robustness to table perturbations")
    session.commit()

    gap_rows = _load_gap_claims(session, [paper.id])
    contribution_rows = _load_contribution_claims(session, [paper.id])
    overlaps = _own_contribution_overlaps(gap_rows, contribution_rows, embedder)

    session.close()
    assert overlaps[gap_rows[0].evidence_id] > 0.5


def test_own_contribution_overlap_is_zero_when_paper_has_no_contribution_claims(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "remains an open problem for future work")
    session.commit()

    gap_rows = _load_gap_claims(session, [paper.id])
    overlaps = _own_contribution_overlaps(gap_rows, [], embedder)

    session.close()
    assert overlaps[gap_rows[0].evidence_id] == 0.0


def test_own_contribution_overlap_only_compares_within_the_same_paper(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, embedder, "a")
    b = _paper(session, embedder, "b")
    _claim(session, a, "research_gap", "no existing benchmark evaluates robustness to table perturbations")
    _claim(session, b, "main_contribution", "we introduce a benchmark evaluating robustness to table perturbations")
    session.commit()

    gap_rows = _load_gap_claims(session, [a.id, b.id])
    contribution_rows = _load_contribution_claims(session, [a.id, b.id])
    overlaps = _own_contribution_overlaps(gap_rows, contribution_rows, embedder)

    session.close()
    # b's near-identical contribution must NOT count toward a's overlap -
    # cross-paper similarity is find_addressing_papers's job (Task 4/7), not this
    assert overlaps[gap_rows[0].evidence_id] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_detect.py -v -k "gap_claims or contribution_claims or overlap"`
Expected: FAIL — names not defined / not importable.

- [ ] **Step 3: Implement in `detect.py`**

Add imports and the new helpers to `src/researchbridge/gaps/detect.py` (keep the existing `_load_claims`/`_render_observation` for now — they get folded into Task 7):

```python
from dataclasses import dataclass
import uuid

from researchbridge.gaps.signals import cosine_similarity

CONTRIBUTION_CLAIM_TYPES = ("main_contribution", "results")


@dataclass
class _GapClaimRow:
    paper_id: uuid.UUID
    evidence_id: uuid.UUID
    text: str
    claim_type: str
    gap_tier: str | None


@dataclass
class _ContributionClaimRow:
    paper_id: uuid.UUID
    text: str


def _load_gap_claims(session: Session, paper_ids: list[uuid.UUID]) -> list[_GapClaimRow]:
    rows = session.execute(
        select(
            ExtractedClaim.paper_id,
            ExtractedClaim.evidence_id,
            ExtractedClaim.text,
            ExtractedClaim.claim_type,
            ExtractedClaim.validation_tier,
        ).where(ExtractedClaim.paper_id.in_(paper_ids), ExtractedClaim.claim_type.in_(RELEVANT_CLAIM_TYPES))
    ).all()
    return [
        _GapClaimRow(paper_id=paper_id, evidence_id=evidence_id, text=text, claim_type=claim_type, gap_tier=tier)
        for paper_id, evidence_id, text, claim_type, tier in rows
    ]


def _load_contribution_claims(session: Session, paper_ids: list[uuid.UUID]) -> list[_ContributionClaimRow]:
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text).where(
            ExtractedClaim.paper_id.in_(paper_ids), ExtractedClaim.claim_type.in_(CONTRIBUTION_CLAIM_TYPES)
        )
    ).all()
    return [_ContributionClaimRow(paper_id=paper_id, text=text) for paper_id, text in rows]


def _own_contribution_overlaps(
    gap_rows: list[_GapClaimRow], contribution_rows: list[_ContributionClaimRow], embedder: Embedder
) -> dict[uuid.UUID, float]:
    """Max cosine similarity between each gap claim and that SAME paper's
    own main_contribution/results claims - 0.0 if the paper has none, and
    never compared against another paper's contribution (that's a separate,
    intentionally distinct check - see find_addressing_papers)."""
    if not gap_rows:
        return {}

    contribution_texts_by_paper: dict[uuid.UUID, list[str]] = {}
    for row in contribution_rows:
        contribution_texts_by_paper.setdefault(row.paper_id, []).append(row.text)

    all_contribution_texts = [row.text for row in contribution_rows]
    gap_texts = [row.text for row in gap_rows]
    vectors = embedder.embed_texts(gap_texts + all_contribution_texts)
    gap_vectors = vectors[: len(gap_texts)]
    contribution_vectors = vectors[len(gap_texts) :]

    vector_by_contribution_text: dict[str, list[float]] = dict(zip(all_contribution_texts, contribution_vectors, strict=True))

    overlaps: dict[uuid.UUID, float] = {}
    for row, vector in zip(gap_rows, gap_vectors, strict=True):
        own_texts = contribution_texts_by_paper.get(row.paper_id, [])
        if not own_texts:
            overlaps[row.evidence_id] = 0.0
            continue
        own_vectors = [vector_by_contribution_text[t] for t in own_texts]
        overlaps[row.evidence_id] = max(cosine_similarity(vector, ov) for ov in own_vectors)

    return overlaps
```

Note: if the same contribution text string happens to repeat verbatim across two different papers, `vector_by_contribution_text` would collide - acceptable for now (identical text implies identical embedding anyway, so the collision is harmless), but leave a one-line comment noting it, matching this codebase's habit of documenting known-acceptable edge cases rather than silently hoping they don't matter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_detect.py -v -k "gap_claims or contribution_claims or overlap"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/detect.py tests/test_gaps_detect.py
git commit -m "feat(gaps): compute per-claim own-contribution overlap in detection"
```

---

## Task 7: `detect.py` — wire classification into `detect_candidate_gaps`, introduce `GapDetectionResult`

**Files:**
- Modify: `src/researchbridge/gaps/detect.py`
- Test: `tests/test_gaps_detect.py` (update existing tests for the new return type)

**Interfaces:**
- Produces: `DetectionStatus = Literal["no_relevant_papers", "insufficient_evidence", "gaps_found"]`
- Produces: `GapDetectionResult` dataclass: `seed_paper_id`, `status: DetectionStatus`, `neighborhood_size: int`, `drafts: list[CandidateGapDraft]`
- Changes: `CandidateGapDraft` gains `gap_status: str`, `resolution_note: str | None`, `evidence_roles: dict[uuid.UUID, ClaimClassification]` (evidence_id → classification, for Task 8's persistence)
- Changes: `detect_candidate_gaps(...) -> GapDetectionResult` (was `-> list[CandidateGapDraft]`) — **breaking change**, consumed by Task 9 (`batch.py`, `cli_detect.py`)

This is the integration point: per-seed status distinguishes "no related papers were even retrieved" from "related papers exist but nothing cleared the evidence bar" from "gaps found" (mirroring `DimensionCoverage`'s `not_assessed` vs `not_found` split), and each surviving cluster gets its `gap_status`/`resolution_note` attached before being returned as a draft. Clusters that `classify_cluster` returns `None` for are dropped here and never become a `CandidateGapDraft` — this is where "fewer, defensible candidates" actually happens.

- [ ] **Step 1: Update the existing tests for the new return shape**

In `tests/test_gaps_detect.py`, every existing call site of `detect_candidate_gaps(...)` needs `drafts = detect_candidate_gaps(...)` changed to `result = detect_candidate_gaps(...)` and downstream references changed from `drafts` to `result.drafts`. Apply this to `test_finds_a_pattern_recurring_across_the_neighborhood`, `test_no_pattern_returns_empty_list`, and `test_observation_names_the_inference_and_is_grounded_in_a_real_claim`. Example for the first:

```python
def test_finds_a_pattern_recurring_across_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    b = _paper(session, embedder, "b")
    c = _paper(session, embedder, "c")

    _claim(session, seed, "limitations", "the system is tested only offline in this setup")
    _claim(session, a, "limitations", "we test the model only offline in our setup")
    _claim(session, b, "research_gap", "testing here happens only offline within this setup")
    _claim(session, c, "limitations", "training requires substantial gpu resources")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].contributing_paper_count == 3
    assert result.drafts[0].seed_paper_id == seed.id
    assert len(result.drafts[0].evidence_ids) == 3
    assert result.drafts[0].gap_status == "known_limitation"  # weak default tier, no strong anchor
```

`test_no_pattern_returns_empty_list` (only 2 papers, below `min_cluster_size`, but `a` is still a real related paper — this is the "relevant papers, insufficient evidence" case, not "no relevant papers"):

```python
def test_no_pattern_returns_empty_list(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")

    _claim(session, seed, "limitations", "the system is tested only offline in this setup")
    _claim(session, a, "limitations", "training requires substantial gpu resources")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []
```

`test_observation_names_the_inference_and_is_grounded_in_a_real_claim`: same `result.drafts[0]` rewrite as the first test, no other logic changes.

Add two new tests for the third status:

```python
def test_no_related_papers_at_all_is_a_distinct_status(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    session.commit()  # no other papers exist at all - nothing for find_similar_to_paper to return

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "no_relevant_papers"
    assert result.drafts == []


def test_neighborhood_size_counts_the_seed_and_its_related_papers(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.neighborhood_size == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_detect.py -v`
Expected: FAIL — `detect_candidate_gaps` still returns a bare list; `GapDetectionResult`/`.status`/`.drafts` don't exist yet.

- [ ] **Step 3: Rewrite `detect_candidate_gaps`**

Replace the body of `src/researchbridge/gaps/detect.py` from the module docstring's `DETECTION_METHOD` line down through the old `_load_claims`/`_render_observation` (keep `_load_gap_claims`/`_load_contribution_claims`/`_own_contribution_overlaps` from Task 6 as-is):

```python
from typing import Literal

from researchbridge.gaps.signals import (
    ClaimClassification,
    ClassifiedMember,
    apply_addressing_downgrade,
    classify_claim,
    classify_cluster,
    cosine_similarity,
    find_addressing_papers,
)

DETECTION_METHOD = "cluster-v2"

DetectionStatus = Literal["no_relevant_papers", "insufficient_evidence", "gaps_found"]


@dataclass
class CandidateGapDraft:
    seed_paper_id: uuid.UUID
    observation: str
    contributing_paper_count: int
    evidence_ids: list[uuid.UUID]
    gap_status: str
    resolution_note: str | None
    evidence_roles: dict[uuid.UUID, ClaimClassification]
    """evidence_id -> its classification within this cluster - persistence.py
    (Task 8) writes these onto CandidateGapEvidence for reviewer provenance."""


@dataclass
class GapDetectionResult:
    seed_paper_id: uuid.UUID
    status: DetectionStatus
    neighborhood_size: int
    """Count of the seed plus every related paper found - "no_relevant_papers"
    means this is 1 (just the seed itself); anything higher but still
    status="insufficient_evidence" means related papers existed but no
    cluster cleared the evidence bar."""
    drafts: list[CandidateGapDraft]


def detect_candidate_gaps(
    session: Session,
    seed_paper_id: uuid.UUID,
    embedder: Embedder,
    top_k: int = 15,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> GapDetectionResult:
    if session.get(Paper, seed_paper_id) is None:
        raise ValueError(f"no paper with id {seed_paper_id}")

    related = find_similar_to_paper(session, seed_paper_id, embedder.model_name, top_k)
    neighborhood_ids = [seed_paper_id] + [paper.id for paper, _distance in related]

    if len(neighborhood_ids) <= 1:
        return GapDetectionResult(
            seed_paper_id=seed_paper_id, status="no_relevant_papers", neighborhood_size=len(neighborhood_ids), drafts=[]
        )

    gap_rows = _load_gap_claims(session, neighborhood_ids)
    contribution_rows = _load_contribution_claims(session, neighborhood_ids)
    overlaps = _own_contribution_overlaps(gap_rows, contribution_rows, embedder)

    classification_by_evidence_id: dict[uuid.UUID, ClaimClassification] = {
        row.evidence_id: classify_claim(row.text, row.claim_type, row.gap_tier, overlaps[row.evidence_id])
        for row in gap_rows
    }
    row_by_evidence_id = {row.evidence_id: row for row in gap_rows}

    claims = [ClaimRecord(paper_id=row.paper_id, evidence_id=row.evidence_id, text=row.text) for row in gap_rows]
    clusters = find_recurring_patterns(claims, embedder, min_cluster_size, similarity_threshold)

    # candidate addressing-signal pool: every contribution/results claim in
    # the neighborhood, embedded once up front rather than per cluster
    contribution_texts = [row.text for row in contribution_rows]
    contribution_vectors = embedder.embed_texts(contribution_texts) if contribution_texts else []
    paper_titles = {p.id: p.title for p, _ in related}
    paper_titles[seed_paper_id] = session.get(Paper, seed_paper_id).title
    addressing_candidates = [
        (row.paper_id, paper_titles.get(row.paper_id, ""), row.text, vector)
        for row, vector in zip(contribution_rows, contribution_vectors, strict=True)
    ]

    drafts: list[CandidateGapDraft] = []
    for cluster in clusters:
        members = [
            ClassifiedMember(
                paper_id=m.paper_id,
                evidence_id=m.evidence_id,
                text=m.text,
                classification=classification_by_evidence_id[m.evidence_id],
            )
            for m in cluster.members
        ]
        status = classify_cluster(members, min_cluster_size)
        if status is None:
            logger.debug(
                "Dropping cluster (insufficient evidence): representative=%r, %d members",
                cluster.representative_text,
                len(cluster.members),
            )
            continue

        [representative_vector] = embedder.embed_texts([cluster.representative_text])
        matches = find_addressing_papers(representative_vector, addressing_candidates)
        final_status, note = apply_addressing_downgrade(status, matches)

        drafts.append(
            CandidateGapDraft(
                seed_paper_id=seed_paper_id,
                observation=_render_observation(cluster),
                contributing_paper_count=cluster.contributing_paper_count,
                evidence_ids=[m.evidence_id for m in cluster.members],
                gap_status=final_status,
                resolution_note=note,
                evidence_roles={m.evidence_id: classification_by_evidence_id[m.evidence_id] for m in cluster.members},
            )
        )

    status: DetectionStatus = "gaps_found" if drafts else "insufficient_evidence"
    return GapDetectionResult(
        seed_paper_id=seed_paper_id, status=status, neighborhood_size=len(neighborhood_ids), drafts=drafts
    )


def _render_observation(cluster) -> str:
    return (
        f"Recurring pattern across {cluster.contributing_paper_count} related papers "
        f"(inference, not stated by any single author): \"{cluster.representative_text}\""
    )
```

Add `import logging` and `logger = logging.getLogger(__name__)` near the top of the module if not already present. Remove the old `_load_claims` function entirely (superseded by `_load_gap_claims`) and update the module-level import list to drop the now-unused `RELEVANT_CLAIM_TYPES` re-export if `_load_gap_claims` (Task 6) already imports it directly — check Task 6's diff before removing anything to avoid a duplicate-import error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_detect.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/detect.py tests/test_gaps_detect.py
git commit -m "feat(gaps): classify clusters and distinguish no-papers/insufficient-evidence/gaps-found"
```

---

## Task 8: `persistence.py` — save gap status and per-evidence provenance

**Files:**
- Modify: `src/researchbridge/gaps/persistence.py`
- Test: `tests/test_gaps_persistence.py`

**Interfaces:**
- Consumes: `CandidateGapDraft.gap_status`/`.resolution_note`/`.evidence_roles` from Task 7

- [ ] **Step 1: Update the failing tests**

The existing `tests/test_gaps_persistence.py` constructs `CandidateGapDraft(...)` directly — every construction needs the 3 new required fields. Update each call site, e.g.:

```python
def test_saves_candidate_gap_as_pending_inference(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2, ev3 = _evidence(session, seed), _evidence(session, seed), _evidence(session, seed)
    session.commit()

    draft = CandidateGapDraft(
        seed_paper_id=seed.id,
        observation='Recurring pattern across 3 related papers (inference): "tested only offline"',
        contributing_paper_count=3,
        evidence_ids=[ev1.id, ev2.id, ev3.id],
        gap_status="known_limitation",
        resolution_note=None,
        evidence_roles={
            ev1.id: ClaimClassification(role="supporting", self_resolution=False, field_scope=False, own_contribution_overlap=0.0),
            ev2.id: ClaimClassification(role="supporting", self_resolution=False, field_scope=False, own_contribution_overlap=0.0),
            ev3.id: ClaimClassification(role="supporting", self_resolution=False, field_scope=False, own_contribution_overlap=0.0),
        },
    )

    saved = save_candidate_gaps(session, [draft], similarity_threshold=0.5)

    assert len(saved) == 1
    gap = session.get(CandidateGap, saved[0].id)
    assert gap.status == "pending"
    assert gap.gap_type == "inference"
    assert gap.contributing_paper_count == 3
    assert gap.similarity_threshold == 0.5
    assert gap.gap_status == "known_limitation"
    assert gap.resolution_note is None
    session.close()
```

Apply the same 3-field addition to `test_links_every_contributing_evidence_row` and `test_saves_multiple_drafts_independently` (any fixed `gap_status`/`resolution_note`/matching `evidence_roles` values work — the point of those two tests isn't gap_status). Add `from researchbridge.gaps.signals import ClaimClassification` to the imports.

Add a new test asserting the per-evidence columns are actually written:

```python
def test_persists_per_evidence_classification(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2 = _evidence(session, seed), _evidence(session, seed)
    session.commit()

    draft = CandidateGapDraft(
        seed_paper_id=seed.id, observation="obs", contributing_paper_count=2, evidence_ids=[ev1.id, ev2.id],
        gap_status="strong_gap", resolution_note="paper X may address this",
        evidence_roles={
            ev1.id: ClaimClassification(role="anchor", self_resolution=False, field_scope=True, own_contribution_overlap=0.1),
            ev2.id: ClaimClassification(role="anchor", self_resolution=False, field_scope=True, own_contribution_overlap=0.2),
        },
    )

    saved = save_candidate_gaps(session, [draft], similarity_threshold=0.4)

    gap = session.get(CandidateGap, saved[0].id)
    assert gap.gap_status == "strong_gap"
    assert gap.resolution_note == "paper X may address this"

    links = {
        link.evidence_id: link
        for link in session.execute(
            select(CandidateGapEvidence).where(CandidateGapEvidence.candidate_gap_id == saved[0].id)
        ).scalars()
    }
    session.close()
    assert links[ev1.id].claim_role == "anchor"
    assert links[ev1.id].field_scope_signal is True
    assert links[ev1.id].own_contribution_overlap == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_persistence.py -v`
Expected: FAIL — `CandidateGapDraft` missing required args; `gap.gap_status`/`link.claim_role` don't exist as writable data yet.

- [ ] **Step 3: Implement**

Rewrite `src/researchbridge/gaps/persistence.py`:

```python
"""Persists CandidateGapDrafts as CandidateGap + CandidateGapEvidence rows.

Always status="pending" - the human-in-the-loop rule for gap detection
isn't optional here. Nothing this module writes should be presented to a
user as a validated finding. gap_status/resolution_note and each evidence
row's classification are written as computed by gaps/detect.py - this
module makes no judgment calls of its own, it's pure persistence.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, CandidateGapEvidence
from researchbridge.gaps.detect import DETECTION_METHOD, CandidateGapDraft


def save_candidate_gaps(
    session: Session, drafts: list[CandidateGapDraft], similarity_threshold: float
) -> list[CandidateGap]:
    saved: list[CandidateGap] = []
    for draft in drafts:
        gap = CandidateGap(
            seed_paper_id=draft.seed_paper_id,
            observation=draft.observation,
            gap_type="inference",
            status="pending",
            contributing_paper_count=draft.contributing_paper_count,
            similarity_threshold=similarity_threshold,
            detection_method=DETECTION_METHOD,
            gap_status=draft.gap_status,
            resolution_note=draft.resolution_note,
        )
        session.add(gap)
        session.flush()  # populate gap.id for the evidence links below

        for evidence_id in draft.evidence_ids:
            classification = draft.evidence_roles[evidence_id]
            session.add(
                CandidateGapEvidence(
                    candidate_gap_id=gap.id,
                    evidence_id=evidence_id,
                    claim_role=classification.role if classification.role != "motivation" else None,
                    self_resolution_signal=classification.self_resolution,
                    field_scope_signal=classification.field_scope,
                    own_contribution_overlap=classification.own_contribution_overlap,
                )
            )
        saved.append(gap)

    session.commit()
    return saved
```

Note the `claim_role if != "motivation" else None`: a saved `CandidateGapDraft`'s `evidence_ids` only ever contains cluster members, and `classify_cluster` only forms clusters from anchor/supporting members in the first place (a `"motivation"`-role claim is excluded from `find_recurring_patterns`'s clustering... actually check this carefully in Task 7's review — `find_recurring_patterns` clusters ALL `RELEVANT_CLAIM_TYPES` claims regardless of role, since role is computed AFTER clustering in Task 7. A `"motivation"`-classified claim CAN still end up as a cluster member if it happened to cluster with others; `classify_cluster`'s role-based counting simply doesn't count it toward `anchor_papers`/`supporting_papers`/`independent_papers`, but it's still in `cluster.members` and therefore still in `evidence_ids`.) This is intentional: a motivation-classified claim can still ride along as evidence *and be visibly labeled as such* (`claim_role=None` renders in the UI as "excluded from corroboration" — Task 12) — the important thing is it never counts toward promoting the cluster's status, and the reviewer can see exactly which contributing quotes counted and which didn't. The DB's CHECK constraint (`claim_role IS NULL OR claim_role IN ('anchor','supporting')`) already only allows those three states, so mapping `"motivation"` to `NULL` here is required, not optional — flag this to whoever reviews the PR since it's a slightly non-obvious mapping.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaps_persistence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/persistence.py tests/test_gaps_persistence.py
git commit -m "feat(gaps): persist gap_status, resolution_note, and per-evidence classification"
```

---

## Task 9: `batch.py` / `cli_detect.py` — adapt to `GapDetectionResult`

**Files:**
- Modify: `src/researchbridge/gaps/batch.py`
- Modify: `src/researchbridge/gaps/cli_detect.py`
- Test: `tests/test_gaps_batch.py`
- Test: `tests/test_gaps_cli_detect.py`

**Interfaces:**
- Consumes: `GapDetectionResult` from Task 7
- Changes: `BatchSummary` gains `no_relevant_papers: int`, `insufficient_evidence: int`

- [ ] **Step 1: Update the failing tests**

In `tests/test_gaps_batch.py`, no call sites construct `GapDetectionResult` directly (they call `run_all`, which calls `detect_candidate_gaps` internally), so most existing tests need no changes — but add new assertions for the new counters:

```python
def test_tracks_no_relevant_papers_separately_from_insufficient_evidence(session_factory, embedder) -> None:
    session = session_factory()
    lonely = _paper(session, embedder, "lonely")  # no other papers exist at all
    session.commit()

    summary = run_all(session, embedder, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.no_relevant_papers == 1
    assert summary.insufficient_evidence == 0


def test_tracks_insufficient_evidence_when_related_papers_exist_but_no_cluster_qualifies(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    _claim(session, seed, "limitations", "the system is tested only offline in this setup")
    _claim(session, a, "limitations", "training requires substantial gpu resources")
    session.commit()

    summary = run_all(session, embedder, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.insufficient_evidence == 1
    assert summary.no_relevant_papers == 0
```

`test_continues_past_a_failure_for_one_paper`'s monkeypatched `_flaky_detect` calls `real_detect(...)` for the non-failing papers and returns its result unmodified — no change needed there since it just forwards whatever `detect_candidate_gaps` returns.

In `tests/test_gaps_cli_detect.py`, check its current contents for any place that inspects `detect_candidate_gaps`'s return value directly (rather than going through `_run_single`/`_run_batch`'s printed output) and update `.drafts` access accordingly — read the file first since its exact test names weren't captured during planning; apply the same `result.drafts`/`result.status` rewrite pattern used in Task 7.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_batch.py tests/test_gaps_cli_detect.py -v`
Expected: FAIL on the two new tests (`no_relevant_papers`/`insufficient_evidence` attributes don't exist on `BatchSummary`); other tests should still pass unmodified at this point since `run_all` hasn't changed yet.

- [ ] **Step 3: Update `batch.py`**

In `src/researchbridge/gaps/batch.py`:

```python
@dataclass
class BatchSummary:
    papers_seen: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    no_relevant_papers: int = 0
    insufficient_evidence: int = 0
    gaps_found: int = 0
    gaps_saved: int = 0
```

```python
    for paper_id in seed_ids:
        summary.papers_seen += 1
        try:
            result = detect_candidate_gaps(session, paper_id, embedder, top_k, min_cluster_size, similarity_threshold)
        except Exception:
            logger.exception("Gap detection failed for paper %s", paper_id)
            summary.papers_failed += 1
            continue

        if result.status == "no_relevant_papers":
            summary.no_relevant_papers += 1
        elif result.status == "insufficient_evidence":
            summary.insufficient_evidence += 1

        summary.gaps_found += len(result.drafts)
        if result.drafts and save:
            saved = save_candidate_gaps(session, result.drafts, similarity_threshold)
            summary.gaps_saved += len(saved)
```

- [ ] **Step 4: Update `cli_detect.py`**

In `_run_single`:

```python
    try:
        result = detect_candidate_gaps(
            session, paper.id, embedder, args.top_k, args.min_cluster_size, args.threshold
        )
    except ValueError as exc:
        print(str(exc))
        return

    print(f"Seed: {paper.title}")
    print(
        f"Neighborhood: top {args.top_k} related papers, "
        f"min cluster size {args.min_cluster_size}, threshold {args.threshold}\n"
    )

    if result.status == "no_relevant_papers":
        print("No related papers were found for this seed - nothing to compare it against.")
        return
    if result.status == "insufficient_evidence":
        print(
            f"Found {result.neighborhood_size - 1} related paper(s), but no cluster of limitation/research_gap "
            "claims cleared the evidence bar - not a failure, just nothing defensible enough to surface."
        )
        return

    for i, draft in enumerate(result.drafts, start=1):
        print(f"[{i}] ({draft.gap_status}) {draft.observation}")
        print(f"    contributing papers: {draft.contributing_paper_count}")
        if draft.resolution_note:
            print(f"    note: {draft.resolution_note}")
        print()

    if args.save:
        saved = save_candidate_gaps(session, result.drafts, args.threshold)
        print(f"Saved {len(saved)} candidate gap(s) with status='pending'.")
```

In `_run_batch`, add the two new summary lines:

```python
    print(f"Papers seen:          {summary.papers_seen}")
    print(f"Papers skipped:       {summary.papers_skipped} (already had a candidate gap)")
    print(f"Papers failed:        {summary.papers_failed}")
    print(f"No related papers:    {summary.no_relevant_papers}")
    print(f"Insufficient evidence:{summary.insufficient_evidence}")
    print(f"Gaps found:           {summary.gaps_found}")
    print(f"Gaps saved:           {summary.gaps_saved}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gaps_batch.py tests/test_gaps_cli_detect.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/gaps/batch.py src/researchbridge/gaps/cli_detect.py tests/test_gaps_batch.py tests/test_gaps_cli_detect.py
git commit -m "feat(gaps): surface no-relevant-papers vs insufficient-evidence in CLI/batch summary"
```

---

## Task 10: Golden regression test set

**Files:**
- Create: `tests/test_gaps_golden_regression.py`

**Interfaces:**
- Consumes: `detect_candidate_gaps`, the fixture helpers already established in `test_gaps_detect.py`/`test_gaps_batch.py`

This is the concrete regression proof for the whole design. Each scenario is its own test, named after the failure class it guards.

- [ ] **Step 1: Write the file**

```python
"""Golden regression set for the gap-evidence-grounding design (see
docs/superpowers/plans/2026-09-01-gap-evidence-grounding.md). Each test
pins down one real or representative false-positive/false-negative pattern
found during manual review of production candidate gaps - a regression in
any of these means the precision-over-recall guarantee has broken.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.gaps.detect import detect_candidate_gaps

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
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


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([f"{title} {source_id}"])
    padded = (vector + [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
    norm = math.sqrt(sum(x * x for x in padded)) or 1.0
    session.add(
        Embedding(
            paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name,
            vector=[x / norm for x in padded],
        )
    )
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, validation_tier: str | None = None) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(
            paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id,
            confidence="medium", validation_tier=validation_tier,
        )
    )


def test_1_paper_motivation_bridging_a_gap_is_excluded(session_factory) -> None:
    """CentaurBench-style: 'this paper investigates whether X can help
    bridge this gap' paired with a near-identical own contribution - both
    self_resolution and own_contribution_overlap fire, so the claim is
    "motivation" and can't seed a cluster."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "centaur seed")
    a = _paper(session, embedder, "a", "centaur a")
    b = _paper(session, embedder, "b", "centaur b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "this paper investigates whether large language models can help bridge this gap in benchmark coverage",
            validation_tier="strong",
        )
        _claim(
            session, paper, "main_contribution",
            "we investigate whether large language models can help bridge this gap in benchmark coverage",
        )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_2_recurring_benchmark_motivation_language_is_excluded(session_factory) -> None:
    """FinRCA-Bench-style: three benchmark papers each honestly motivate
    their own contribution with near-identical 'existing benchmarks lack X'
    phrasing, where X is exactly what their own contribution measures. No
    self-resolution phrase in the limitation sentence itself, but the
    overlap with each paper's OWN contribution is what excludes it."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "finrca seed")
    a = _paper(session, embedder, "a", "finrca a")
    b = _paper(session, embedder, "b", "finrca b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "no existing benchmark evaluates robustness to table perturbations and reasoning pathway aggregation",
            validation_tier="strong",
        )
        _claim(
            session, paper, "main_contribution",
            "we introduce a benchmark evaluating robustness to table perturbations and reasoning pathway aggregation",
        )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_3_contribution_statement_mistaken_for_a_gap_is_excluded(session_factory) -> None:
    """A results-flavored sentence should never even reach ExtractedClaim
    typed as research_gap - extraction/validation.py's existing gate
    (unchanged by this plan) is what stops it. Regression-proofs that this
    plan didn't weaken it."""
    from researchbridge.extraction.pipeline import ExtractionPipeline
    from researchbridge.extraction.base import ClaimCandidate

    embedder = WordOverlapEmbedder()
    session = session_factory()
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    paper = Paper(
        source="fake", source_id="contrib-1", title="t", abstract=text, raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()

    class _Extractor:
        extraction_method = "fake"
        model_version = "fake-v1"

        def extract(self, p):
            return [ClaimCandidate(claim_type="research_gap", claim_text=text, evidence_quote=text, confidence="medium")]

    ExtractionPipeline(_Extractor(), session_factory).run()

    claims = session.execute(
        __import__("sqlalchemy").select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)
    ).scalars().all()
    session.close()
    assert claims == []  # rejected at extraction validation, never reaches the gap pipeline at all


def test_4_shared_topic_with_no_unresolvedness_evidence_is_excluded(session_factory) -> None:
    """Three papers sharing a research topic (typed as 'problem', which is
    outside RELEVANT_CLAIM_TYPES) never enter gap clustering in the first
    place - there are related papers, but zero limitation/research_gap
    claims among them at all."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "topic seed")
    a = _paper(session, embedder, "a", "topic a")
    b = _paper(session, embedder, "b", "topic b")
    for paper in (seed, a, b):
        _claim(session, paper, "problem", "long-context reasoning in large language models is challenging")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_5_explicit_field_level_unresolved_gap_is_a_strong_gap(session_factory) -> None:
    """The positive case: two independent papers state unambiguous,
    field-scoped unresolved-gap language, with no self-resolution and no
    overlap with their own (unrelated) contributions."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "strong seed")
    a = _paper(session, embedder, "a", "strong a")
    b = _paper(session, embedder, "b", "strong b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "despite recent progress robustness to adversarial perturbations remains unresolved for existing methods",
            validation_tier="strong",
        )
        _claim(session, paper, "main_contribution", "we propose a transformer based text classification architecture")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].gap_status == "strong_gap"


def test_6_genuine_recurring_limitation_without_gap_vocabulary_is_known_limitation(session_factory) -> None:
    """Three papers each admit the same residual weakness in plain
    limitation language (not the stronger research_gap vocabulary), with no
    overlap against their own unrelated contributions - a real, if lower-
    confidence, candidate."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "limitation seed")
    a = _paper(session, embedder, "a", "limitation a")
    b = _paper(session, embedder, "b", "limitation b")
    for paper in (seed, a, b):
        _claim(session, paper, "limitations", "however the approach does not scale to graphs with a million nodes")
        _claim(session, paper, "main_contribution", "we propose a lock free queue for high throughput systems")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].gap_status == "known_limitation"


def test_7_benchmark_paper_similar_to_but_not_solving_the_gap_still_corroborates(session_factory) -> None:
    """Two independent papers anchor a strong_gap; a third paper only
    benchmarks/evaluates the same problem (its own contribution overlaps
    heavily, but it never uses self-resolution language) - it must be
    demoted to supporting, not dropped, and must not block the strong_gap
    the other two already earned."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "mixed seed")
    a = _paper(session, embedder, "a", "mixed a")
    benchmark_paper = _paper(session, embedder, "bench", "mixed bench")

    for paper in (seed, a):
        _claim(
            session, paper, "research_gap",
            "despite recent progress evaluation under adversarial perturbations remains unresolved for existing benchmarks",
            validation_tier="strong",
        )
        _claim(session, paper, "main_contribution", "we propose a transformer based text classification architecture")

    _claim(
        session, benchmark_paper, "research_gap",
        "no existing benchmark evaluates adversarial perturbations under realistic constraints for existing benchmarks",
        validation_tier="strong",
    )
    _claim(
        session, benchmark_paper, "main_contribution",
        "we introduce a benchmark for evaluating adversarial perturbations under realistic constraints",
    )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    draft = result.drafts[0]
    assert draft.gap_status == "strong_gap"
    assert draft.contributing_paper_count == 3  # the benchmark paper's evidence still shows up...
    benchmark_classification = next(
        c for eid, c in draft.evidence_roles.items() if c.own_contribution_overlap >= 0.5
    )
    assert benchmark_classification.role == "supporting"  # ...but demoted, not counted as an anchor
```

Note on wording: the shared vocabulary ("existing benchmarks"/"existing methods") deliberately overlaps across test 5/6/7's sentences so `WordOverlapEmbedder`'s bag-of-words similarity clusters them the way real embedding similarity would for genuinely related sentences — this mirrors the existing `test_gaps_cluster.py`/`test_gaps_detect.py` convention of reusing a repeated word (e.g. "offline") to force a predictable fake-embedder cluster. If any of these tests' cluster doesn't form as expected under `WordOverlapEmbedder` (check via a quick manual run before trusting the assertions), adjust the shared wording to increase bag-of-words overlap within each scenario's 2-3 sentences without changing which regex/overlap signals they're meant to exercise.

- [ ] **Step 2: Run the tests, iterate on wording until each passes for the right reason**

Run: `pytest tests/test_gaps_golden_regression.py -v`
Expected: All 7 pass. If a clustering assertion fails because `WordOverlapEmbedder`'s bag-of-words similarity didn't group the intended sentences, adjust shared vocabulary (not the assertions) until it does — the fake embedder is a blunt instrument and needs literal word overlap, unlike the real `SentenceTransformerEmbedder` these scenarios are modeling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gaps_golden_regression.py
git commit -m "test(gaps): add golden regression set for the 7 false-positive/negative patterns"
```

---

## Task 11: API — expose `gap_status`, `resolution_note`, and per-evidence provenance

**Files:**
- Modify: `src/researchbridge/api/schemas.py:114-146`
- Modify: `src/researchbridge/api/serializers.py:110-202`
- Test: `tests/test_gaps_api.py`

**Interfaces:**
- Consumes: `CandidateGap.gap_status`/`.resolution_note`, `CandidateGapEvidence.claim_role`/`.self_resolution_signal`/`.field_scope_signal`/`.own_contribution_overlap`, `ExtractedClaim.claim_type`/`.validation_tier` (joined via `Evidence.id == ExtractedClaim.evidence_id`)

- [ ] **Step 1: Write the failing tests**

Check `tests/test_gaps_api.py` for its existing fixture pattern (likely similar `_paper`/`_evidence`/API-client-via-`TestClient` helpers) before writing — match its style. Add:

```python
def test_gap_response_includes_status_and_resolution_note(client, session_factory) -> None:
    # ... build a CandidateGap with gap_status="strong_gap", resolution_note="note text" via
    # the same fixture helpers the rest of the file uses, then GET /api/gaps
    response = client.get("/api/gaps?status=pending")
    body = response.json()
    gap = body["items"][0]
    assert gap["gap_status"] == "strong_gap"
    assert gap["resolution_note"] == "note text"


def test_gap_evidence_includes_claim_type_and_role(client, session_factory) -> None:
    response = client.get("/api/gaps?status=pending")
    body = response.json()
    evidence = body["items"][0]["evidence"][0]
    assert evidence["claim_type"] in {"limitations", "research_gap"}
    assert evidence["claim_role"] in {"anchor", "supporting", None}
    assert "validation_tier" in evidence
    assert "self_resolution_signal" in evidence
    assert "field_scope_signal" in evidence
    assert "own_contribution_overlap" in evidence
```

Write these against the file's actual existing fixtures (read `tests/test_gaps_api.py` in full first — it wasn't captured verbatim during planning) rather than inventing a different setup pattern; the goal is these two tests read like siblings of the file's existing tests, using the same DB-seeding helpers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_api.py -v -k "status_and_resolution or claim_type_and_role"`
Expected: FAIL — `KeyError`/`None` on the new fields.

- [ ] **Step 3: Update `schemas.py`**

```python
class GapEvidenceOut(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    text: str
    section: str | None
    claim_type: str
    validation_tier: str | None
    """"strong"/"weak" for a research_gap claim, None for limitations or any
    claim predating this column (see extraction/validation.py)."""
    claim_role: str | None
    """"anchor"/"supporting" - this evidence's role within its cluster. None
    means it was classified "motivation" (excluded from corroborating the
    gap, but still shown for transparency - see gaps/persistence.py)."""
    self_resolution_signal: bool
    field_scope_signal: bool
    own_contribution_overlap: float

    model_config = {"from_attributes": True}


class CandidateGapOut(BaseModel):
    id: uuid.UUID
    seed_paper_id: uuid.UUID
    seed_paper_title: str
    observation: str
    gap_type: str
    status: str
    gap_status: str | None
    resolution_note: str | None
    contributing_paper_count: int
    similarity_threshold: float
    detection_method: str
    review_note: str | None
    correctness_rating: int | None
    relevance_rating: int | None
    novelty_rating: int | None
    evidence_support_rating: int | None
    usefulness_rating: int | None
    evidence: list[GapEvidenceOut]

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Update `serializers.py`**

```python
def to_gaps(session: Session, gaps: Sequence[CandidateGap]) -> list[CandidateGapOut]:
    if not gaps:
        return []

    gap_ids = [g.id for g in gaps]
    seed_ids = {g.seed_paper_id for g in gaps}

    titles_by_paper_id = _titles_by_paper(session, seed_ids)
    evidence_by_gap = _evidence_by_gap(session, gap_ids)

    return [
        CandidateGapOut(
            id=gap.id,
            seed_paper_id=gap.seed_paper_id,
            seed_paper_title=titles_by_paper_id.get(gap.seed_paper_id, ""),
            observation=gap.observation,
            gap_type=gap.gap_type,
            status=gap.status,
            gap_status=gap.gap_status,
            resolution_note=gap.resolution_note,
            contributing_paper_count=gap.contributing_paper_count,
            similarity_threshold=gap.similarity_threshold,
            detection_method=gap.detection_method,
            review_note=gap.review_note,
            correctness_rating=gap.correctness_rating,
            relevance_rating=gap.relevance_rating,
            novelty_rating=gap.novelty_rating,
            evidence_support_rating=gap.evidence_support_rating,
            usefulness_rating=gap.usefulness_rating,
            evidence=evidence_by_gap.get(gap.id, []),
        )
        for gap in gaps
    ]
```

```python
def _evidence_by_gap(session: Session, gap_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[GapEvidenceOut]]:
    rows = session.execute(
        select(CandidateGapEvidence, Evidence, Paper.title, ExtractedClaim.validation_tier)
        .join(Evidence, Evidence.id == CandidateGapEvidence.evidence_id)
        .join(Paper, Paper.id == Evidence.paper_id)
        .join(ExtractedClaim, ExtractedClaim.evidence_id == Evidence.id)
        .where(CandidateGapEvidence.candidate_gap_id.in_(gap_ids))
    ).all()

    result: dict[uuid.UUID, list[GapEvidenceOut]] = defaultdict(list)
    for link, evidence, paper_title, validation_tier in rows:
        result[link.candidate_gap_id].append(
            GapEvidenceOut(
                paper_id=evidence.paper_id,
                paper_title=paper_title,
                text=evidence.text,
                section=evidence.section,
                claim_type=evidence.evidence_type,
                validation_tier=validation_tier,
                claim_role=link.claim_role,
                self_resolution_signal=bool(link.self_resolution_signal),
                field_scope_signal=bool(link.field_scope_signal),
                own_contribution_overlap=link.own_contribution_overlap or 0.0,
            )
        )
    return result
```

Add `ExtractedClaim` to the `from researchbridge.db.models import (...)` block at the top of `serializers.py` if not already imported.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gaps_api.py -v`
Expected: PASS (full file, not just the two new tests — confirm nothing else regressed)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/serializers.py tests/test_gaps_api.py
git commit -m "feat(api): expose gap_status, resolution_note, and per-evidence claim provenance"
```

---

## Task 12: Frontend — status badges, per-evidence labels, resolution note

**Files:**
- Modify: `frontend/lib/gapsApi.ts`
- Modify: `frontend/app/gaps/page.tsx`

- [ ] **Step 1: Update the TypeScript types**

In `frontend/lib/gapsApi.ts`:

```typescript
export type GapEvidence = {
  paper_id: string;
  paper_title: string;
  text: string;
  section: string | null;
  claim_type: string;
  validation_tier: "strong" | "weak" | null;
  claim_role: "anchor" | "supporting" | null;
  self_resolution_signal: boolean;
  field_scope_signal: boolean;
  own_contribution_overlap: number;
};
```

```typescript
export type CandidateGap = GapRatings & {
  id: string;
  seed_paper_id: string;
  seed_paper_title: string;
  observation: string;
  gap_type: string;
  status: "pending" | "approved" | "rejected";
  gap_status: "strong_gap" | "potential_gap" | "known_limitation" | null;
  resolution_note: string | null;
  contributing_paper_count: number;
  similarity_threshold: number;
  detection_method: string;
  review_note: string | null;
  evidence: GapEvidence[];
};
```

Add a small display-label lookup near `RATING_DIMENSIONS`:

```typescript
export const GAP_STATUS_LABELS: Record<NonNullable<CandidateGap["gap_status"]>, string> = {
  strong_gap: "strong gap",
  potential_gap: "potential gap",
  known_limitation: "known limitation",
};
```

- [ ] **Step 2: Render the status badge and resolution note in `GapCard`**

In `frontend/app/gaps/page.tsx`, import `GAP_STATUS_LABELS` alongside the existing `RATING_DIMENSIONS` import, then add a badge next to the existing `gap.contributing_paper_count · threshold ...` line (around line 190-193):

```tsx
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <Link href={`/papers/${gap.seed_paper_id}`} className="eyebrow hover:text-[var(--ink)]">
          seed: {gap.seed_paper_title}
        </Link>
        <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
          {gap.contributing_paper_count} papers · threshold {gap.similarity_threshold.toFixed(2)} ·{" "}
          {gap.detection_method}
        </span>
      </div>

      {gap.gap_status && (
        <span
          className={`eyebrow mt-2 inline-block rounded-[2px] border px-2 py-0.5 text-[0.6875rem] ${
            gap.gap_status === "strong_gap"
              ? "border-[var(--ink)] text-[var(--ink)]"
              : gap.gap_status === "potential_gap"
                ? "border-[var(--rule)] text-[var(--ink-soft)]"
                : "border-[var(--rule-soft)] text-[var(--ink-faint)]"
          }`}
        >
          {GAP_STATUS_LABELS[gap.gap_status]}
        </span>
      )}
```

Add the resolution note callout right after the existing `gap.gap_type` line (around line 200-202):

```tsx
      <div className="mt-2 text-[0.75rem] text-[var(--ink-faint)]">
        {gap.gap_type} — a synthesized observation, not a stated fact
      </div>

      {gap.resolution_note && (
        <p className="mt-2 max-w-[68ch] text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
          ⚠ {gap.resolution_note}
        </p>
      )}
```

- [ ] **Step 3: Label each evidence quote with its claim type and role**

In the evidence `<li>` block (around line 211-220), add the claim-type/role badges next to the paper link:

```tsx
            {gap.evidence.map((ev, i) => (
              <li key={i}>
                <div className="flex flex-wrap items-center gap-2">
                  <Link href={`/papers/${ev.paper_id}`} className="eyebrow hover:text-[var(--ink)]">
                    {ev.paper_title}
                  </Link>
                  <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                    {ev.claim_type}
                    {ev.validation_tier ? ` · ${ev.validation_tier}` : ""}
                    {ev.claim_role ? ` · ${ev.claim_role}` : " · excluded from corroboration"}
                  </span>
                </div>
                <p className="mt-1 max-w-[62ch] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                  &quot;{ev.text}&quot;
                </p>
              </li>
            ))}
```

- [ ] **Step 4: Verify in the browser**

Run: use `preview_start` with the frontend's dev server config from `.claude/launch.json` (create one if it doesn't exist, matching the pattern in the tool's own instructions — `npm run dev` in `frontend/`, whatever port `frontend/package.json`'s dev script uses).

Navigate to `/gaps`, confirm: the status badge renders next to each pending candidate, the evidence list shows claim type/tier/role per quote, and a resolution note (if any test data has one) renders with the warning glyph. Check both light and dark theme via `resize_window`'s `colorScheme` if the app supports a theme toggle — check `frontend/app/layout.tsx`/`globals.css` first for how `--ink`/`--rule` etc. are themed before assuming dark mode needs separate handling; these badges reuse existing CSS variables so no new theme-specific work should be needed, but confirm rather than assume.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/gapsApi.ts frontend/app/gaps/page.tsx
git commit -m "feat(frontend): show gap status, resolution note, and per-evidence claim provenance"
```

---

## Task 13: Build the threshold-calibration evaluation harness

**Files:**
- Create: `src/researchbridge/gaps/calibration.py`
- Create: `src/researchbridge/gaps/cli_calibrate.py`
- Create: `benchmark/gap_calibration_groups.yaml`
- Test: `tests/test_gaps_calibration.py`

**Interfaces:**
- Produces: `ThresholdSample` dataclass, `own_contribution_overlap_samples(annotations, embedder) -> list[ThresholdSample]`, `own_contribution_overlap_group_samples(annotations, groups, embedder) -> list[ThresholdSample]`, `addressing_signal_samples(annotations, groups, embedder) -> list[ThresholdSample]`, `ThresholdSweepRow` dataclass, `sweep_thresholds(samples, thresholds) -> list[ThresholdSweepRow]`, `CalibrationGroup` dataclass, `load_calibration_groups(path) -> list[CalibrationGroup]`
- Consumed by: Task 14 (the real-corpus run)

**These two thresholds measure different things and are calibrated as two separate, independent evaluations — not one combined pass.** `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` has a comparatively clear interpretation (does this paper's own contribution overlap with the gap/limitation it itself states?), so its calibration samples are scored as "this pair should/shouldn't overlap." `ADDRESSING_SIMILARITY_THRESHOLD` has a deliberately weaker interpretation: high similarity between Paper A's gap and Paper B's (a *different* paper's) contribution never establishes that B solved or fully addressed A's gap — B might be a benchmark, an evaluation, a partial solution, a dataset, or unrelated infrastructure that merely shares vocabulary. Its calibration samples are therefore scored as "should this pair trigger a reviewer warning," never as "should this pair be treated as resolution." `find_addressing_papers`/`apply_addressing_downgrade` (Task 4) are unchanged by this — they already only ever downgrade and note, never invalidate; this task only makes sure the calibration language and category set match that design instead of overclaiming.

`OWN_CONTRIBUTION_OVERLAP_THRESHOLD` and `ADDRESSING_SIMILARITY_THRESHOLD` were set to `0.5` as a starting point in Tasks 2 and 4. This task builds the actual measurement instrument before either constant is treated as final — modeled directly on two evaluation patterns already established in this codebase: `extraction/type_validation_evaluation.py`'s own-field-acceptance/cross-field-rejection split (the closest existing analog to "does a real ground-truth pair correctly get flagged or not"), and `gaps/cluster.py`'s own calibration-note precedent of reporting a measured distribution rather than a guess.

**`OWN_CONTRIBUTION_OVERLAP_THRESHOLD` categories** (same-paper: a claim's text against that *same* paper's own contribution):
- **genuine unresolved gap** (`genuine_unresolved_gap`): `research_gap.remaining` vs `main_contribution`, straight from the Sec 25 annotations — the annotator's explicit statement of what's still open, paired against that same paper's contribution. Real "should NOT overlap" ground truth, no new annotation effort.
- **motivation / contribution statements** (`motivation_addressed`): `research_gap.addressed` vs `main_contribution`, also straight from the annotations — the annotator's explicit statement of what the paper's own contribution already resolved. Real "SHOULD overlap" ground truth.
- **recurring limitations genuinely unresolved** (`recurring_limitation_genuinely_unresolved`) and **recurring limitations that are merely topical convergence** (`recurring_limitation_topical_convergence`): these still need a curated `benchmark/gap_calibration_groups.yaml` group to *find* real examples in the corpus (a thematically-connected cluster of same-subfield papers), but each computed sample is still a same-paper self-pair — paper A's own limitation text against paper A's own contribution — exactly what `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` gates in `classify_cluster`'s `independent_papers` filter (Task 3). This is the FinRCA-Bench pattern: it's driven by each paper's own overlap, not by comparing across different papers, so it belongs to this threshold, not the addressing signal below.

**`ADDRESSING_SIMILARITY_THRESHOLD` categories** (cross-paper: Paper A's gap text against a *different* Paper B's contribution) — all three come from curated `benchmark/gap_calibration_groups.yaml` groups, since the per-paper annotations have no cross-paper structure:
- **high-value addressing-review matches** (`high_value_addressing_match`): Paper B's contribution plausibly bears on Paper A's stated gap — a real case where the warning is worth showing a reviewer. Scored as "should trigger a warning."
- **false warning cases** (`false_warning_topical_only`): Paper A and Paper B share only topical/domain vocabulary, with no real bearing on each other's gap — a case where a warning would be noise. Scored as "should NOT trigger a warning."
- **benchmark/evaluation/partial-solution cases** (`benchmark_evaluation_partial_solution`): Paper B's contribution is a benchmark, evaluation, dataset, or partial solution touching Paper A's stated gap, without solving the underlying problem — this SHOULD still trigger a warning (that's the whole point of the signal), but is never interpreted as proof of resolution. Golden test #7 (Task 10) is the runtime-behavior twin of this category: the corresponding evidence stays `supporting`, and the cluster's `strong_gap` status stands.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gaps_calibration.py`:

```python
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from researchbridge.benchmark.store import Annotation
from researchbridge.gaps.calibration import (
    CalibrationGroup,
    ThresholdSample,
    addressing_signal_samples,
    load_calibration_groups,
    own_contribution_overlap_group_samples,
    own_contribution_overlap_samples,
    sweep_thresholds,
)

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
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


def _annotation(source_id: str, contribution: str, remaining: str = "", addressed: str = "") -> Annotation:
    return Annotation(
        source_id=source_id,
        path=None,  # not read by these functions
        identity={"domain": "Machine Learning"},
        fields={"main_contribution": contribution},
        research_gap={"remaining": remaining, "addressed": addressed},
    )


def test_own_contribution_overlap_samples_pairs_remaining_against_contribution() -> None:
    annotations = [
        _annotation("p1", contribution="we build a robustness benchmark", remaining="fairness across languages is untested")
    ]

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    remaining_samples = [s for s in samples if s.category == "genuine_unresolved_gap"]
    assert len(remaining_samples) == 1
    assert remaining_samples[0].expected_high_overlap is False
    assert remaining_samples[0].paper_a == "p1"


def test_own_contribution_overlap_samples_pairs_addressed_against_contribution() -> None:
    annotations = [
        _annotation("p1", contribution="we build a robustness benchmark", addressed="prior work lacked a robustness benchmark")
    ]

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    addressed_samples = [s for s in samples if s.category == "motivation_addressed"]
    assert len(addressed_samples) == 1
    assert addressed_samples[0].expected_high_overlap is True


def test_own_contribution_overlap_samples_skips_blank_fields() -> None:
    annotations = [_annotation("p1", contribution="we build a benchmark")]  # no remaining/addressed text

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    assert samples == []


def test_sweep_thresholds_scores_each_category_at_each_threshold() -> None:
    samples = [
        ThresholdSample("genuine_unresolved_gap", "p1", "p1", "a", "b", similarity=0.2, expected_high_overlap=False),
        ThresholdSample("genuine_unresolved_gap", "p2", "p2", "a", "b", similarity=0.6, expected_high_overlap=False),
        ThresholdSample("motivation_addressed", "p1", "p1", "a", "b", similarity=0.7, expected_high_overlap=True),
    ]

    rows = sweep_thresholds(samples, thresholds=[0.5])

    by_category = {r.category: r for r in rows}
    # genuine_unresolved_gap: one sample (0.6) incorrectly clears 0.5 - a false positive (a real gap
    # would get wrongly downgraded)
    assert by_category["genuine_unresolved_gap"].false_positive_rate == 0.5
    # motivation_addressed: the one sample (0.7) correctly clears 0.5
    assert by_category["motivation_addressed"].false_negative_rate == 0.0


def test_load_calibration_groups_reads_the_yaml_schema(tmp_path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text(
        "groups:\n"
        "  - category: recurring_limitation_topical_convergence\n"
        "    source_ids: ['1111.11111', '2222.22222']\n"
        "    note: shared benchmark motivation\n"
    )

    groups = load_calibration_groups(path)

    assert len(groups) == 1
    assert groups[0].category == "recurring_limitation_topical_convergence"
    assert groups[0].source_ids == ["1111.11111", "2222.22222"]


def test_load_calibration_groups_missing_file_returns_empty() -> None:
    groups = load_calibration_groups("/does/not/exist.yaml")

    assert groups == []


def test_own_contribution_overlap_group_samples_pairs_each_member_against_itself() -> None:
    # recurring_limitation_* categories still need a curated group to FIND real
    # examples, but each sample is a same-paper self-pair - this calibrates
    # OWN_CONTRIBUTION_OVERLAP_THRESHOLD, not the cross-paper addressing signal
    annotations = [
        _annotation("p1", contribution="we introduce a benchmark for table robustness", remaining="no existing benchmark evaluates table robustness"),
        _annotation("p2", contribution="we introduce a benchmark for reasoning aggregation", remaining="no existing benchmark evaluates reasoning aggregation"),
    ]
    groups = [
        CalibrationGroup(category="recurring_limitation_topical_convergence", source_ids=["p1", "p2"], note="benchmark motivation")
    ]

    samples = own_contribution_overlap_group_samples(annotations, groups, WordOverlapEmbedder())

    assert len(samples) == 2  # one self-pair per member, not one pair across members
    assert {s.paper_a for s in samples} == {"p1", "p2"}
    assert all(s.paper_a == s.paper_b for s in samples)  # same-paper, not cross-paper
    assert all(s.expected_high_overlap is True for s in samples)


def test_own_contribution_overlap_group_samples_unknown_category_is_ignored() -> None:
    annotations = [_annotation("p1", contribution="c", remaining="r")]
    groups = [CalibrationGroup(category="high_value_addressing_match", source_ids=["p1"], note="")]

    samples = own_contribution_overlap_group_samples(annotations, groups, WordOverlapEmbedder())

    assert samples == []  # that category belongs to addressing_signal_samples, not this function


def test_addressing_signal_samples_needs_at_least_two_source_ids_per_group() -> None:
    annotations = [_annotation("solo", contribution="we build a benchmark")]
    groups = [CalibrationGroup(category="high_value_addressing_match", source_ids=["solo"], note="")]

    samples = addressing_signal_samples(annotations, groups, WordOverlapEmbedder())

    assert samples == []  # nothing to compare a lone paper against within its own group


def test_addressing_signal_samples_compares_across_different_papers_only() -> None:
    annotations = [
        _annotation("a", contribution="unrelated method", remaining="robustness to table perturbations is untested"),
        _annotation("b", contribution="we build a benchmark evaluating robustness to table perturbations"),
    ]
    groups = [CalibrationGroup(category="benchmark_evaluation_partial_solution", source_ids=["a", "b"], note="")]

    samples = addressing_signal_samples(annotations, groups, WordOverlapEmbedder())

    assert len(samples) == 1
    assert samples[0].paper_a == "a"
    assert samples[0].paper_b == "b"  # cross-paper: A's gap text against B's (different paper) contribution
    assert samples[0].expected_high_overlap is True  # should warn - never interpreted as "B solved it"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaps_calibration.py -v`
Expected: FAIL — `researchbridge.gaps.calibration` doesn't exist yet.

- [ ] **Step 3: Implement `gaps/calibration.py`**

```python
"""Calibrates OWN_CONTRIBUTION_OVERLAP_THRESHOLD and ADDRESSING_SIMILARITY_THRESHOLD
(gaps/signals.py) against real, hand-labeled data before either constant is
treated as final - see docs/superpowers/plans/2026-09-01-gap-evidence-
grounding.md Tasks 13-14.

Mirrors extraction/type_validation_evaluation.py's own-field/cross-field
pattern: measure real ground-truth pairs that SHOULD overlap against real
pairs that should NOT, rather than eyeballing a single number.

Two data sources, covering all six required calibration categories:

1. The Sec 25 hand-annotated benchmark (benchmark.store.load_all) already
   contains real ground truth for two categories, no new annotation
   needed: research_gap.remaining vs main_contribution is a real "should
   NOT overlap" pair (genuine_unresolved_gap); research_gap.addressed vs
   main_contribution is a real "SHOULD overlap" pair (motivation_addressed).

2. benchmark/gap_calibration_groups.yaml, a small hand-curated companion
   file, for the categories that need a curated group of thematically-
   connected real papers to find examples of.

The two thresholds this calibrates measure different things and are never
scored together: OWN_CONTRIBUTION_OVERLAP_THRESHOLD asks "does this claim
overlap with what its OWN paper contributed" (own_contribution_overlap_
samples, own_contribution_overlap_group_samples) - a comparatively clear
question. ADDRESSING_SIMILARITY_THRESHOLD asks "is this claim similar
enough to a DIFFERENT paper's contribution to be worth a reviewer warning"
(addressing_signal_samples) - a deliberately weaker question, since high
similarity to another paper's benchmark/evaluation/partial-solution
contribution is never proof that paper resolved the gap. Mixing the two
into one pool would blur two different failure costs: a false positive on
the own-contribution side wrongly excludes real evidence, while a false
positive on the addressing side just erodes trust in an already-advisory
warning - see Task 14's separate decision rules for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from researchbridge.benchmark.store import Annotation
from researchbridge.embedding.base import Embedder
from researchbridge.gaps.signals import cosine_similarity


@dataclass(frozen=True)
class ThresholdSample:
    category: str
    paper_a: str
    paper_b: str
    text_a: str
    text_b: str
    similarity: float
    expected_high_overlap: bool
    """Ground truth: True if this pair SHOULD clear the threshold (a real
    motivation/addressed/benchmark-not-solving pair), False if it should
    NOT (a real genuine-unresolved-gap or topically-unrelated pair)."""


@dataclass(frozen=True)
class CalibrationGroup:
    category: str
    source_ids: list[str]
    note: str


def load_calibration_groups(path: str | Path) -> list[CalibrationGroup]:
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        CalibrationGroup(category=g["category"], source_ids=list(g["source_ids"]), note=g.get("note", ""))
        for g in raw.get("groups", [])
    ]


def own_contribution_overlap_samples(annotations: list[Annotation], embedder: Embedder) -> list[ThresholdSample]:
    texts: list[str] = []
    meta: list[tuple[str, str, str, str, bool]] = []
    for ann in annotations:
        contribution = (ann.fields.get("main_contribution") or "").strip()
        remaining = (ann.research_gap.get("remaining") or "").strip()
        addressed = (ann.research_gap.get("addressed") or "").strip()
        if contribution and remaining:
            texts += [contribution, remaining]
            meta.append(("genuine_unresolved_gap", ann.source_id, contribution, remaining, False))
        if contribution and addressed:
            texts += [contribution, addressed]
            meta.append(("motivation_addressed", ann.source_id, contribution, addressed, True))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, source_id, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=source_id, paper_b=source_id,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


# These two categories still need a curated GROUP to find real examples of
# in the corpus, but each computed sample is a same-paper self-pair (this
# paper's own limitation text against this SAME paper's own contribution) -
# calibrates OWN_CONTRIBUTION_OVERLAP_THRESHOLD, same as
# own_contribution_overlap_samples above, NOT the cross-paper addressing
# signal below. See classify_cluster's independent_papers filter
# (gaps/signals.py) - this is exactly what gates that.
_OWN_OVERLAP_GROUP_EXPECTATIONS = {
    "recurring_limitation_genuinely_unresolved": False,
    "recurring_limitation_topical_convergence": True,
}


def own_contribution_overlap_group_samples(
    annotations: list[Annotation], groups: list[CalibrationGroup], embedder: Embedder
) -> list[ThresholdSample]:
    annotations_by_source_id = {a.source_id: a for a in annotations}

    texts: list[str] = []
    meta: list[tuple[str, str, str, str, bool]] = []
    for group in groups:
        expected = _OWN_OVERLAP_GROUP_EXPECTATIONS.get(group.category)
        if expected is None:
            continue
        for source_id in group.source_ids:
            ann = annotations_by_source_id.get(source_id)
            if ann is None:
                continue
            contribution = (ann.fields.get("main_contribution") or "").strip()
            limitation_text = (ann.research_gap.get("remaining") or ann.fields.get("limitations") or "").strip()
            if not contribution or not limitation_text:
                continue
            texts += [limitation_text, contribution]
            meta.append((group.category, source_id, limitation_text, contribution, expected))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, source_id, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=source_id, paper_b=source_id,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


# Cross-paper only: does Paper A's gap text warrant a reviewer warning
# against Paper B's (a DIFFERENT paper's) contribution? expected_high_overlap
# here means "should trigger a warning" - it is NEVER interpreted as "B
# solved A's gap". benchmark_evaluation_partial_solution is the category
# that makes this explicit: it expects a warning (True) precisely because a
# benchmark/evaluation/partial-solution contribution is exactly the case
# apply_addressing_downgrade exists to flag, while still leaving the
# cluster's status a downgrade, never an invalidation (see Task 4, Task 10
# golden test 7).
_ADDRESSING_GROUP_EXPECTATIONS = {
    "high_value_addressing_match": True,
    "false_warning_topical_only": False,
    "benchmark_evaluation_partial_solution": True,
}


def addressing_signal_samples(
    annotations: list[Annotation], groups: list[CalibrationGroup], embedder: Embedder
) -> list[ThresholdSample]:
    annotations_by_source_id = {a.source_id: a for a in annotations}

    texts: list[str] = []
    meta: list[tuple[str, str, str, str, str, bool]] = []
    for group in groups:
        expected = _ADDRESSING_GROUP_EXPECTATIONS.get(group.category)
        if expected is None or len(group.source_ids) < 2:
            continue
        members = [annotations_by_source_id[sid] for sid in group.source_ids if sid in annotations_by_source_id]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                text_a = (a.research_gap.get("remaining") or a.fields.get("limitations") or "").strip()
                text_b = (b.fields.get("main_contribution") or "").strip()
                if not text_a or not text_b:
                    continue
                texts += [text_a, text_b]
                meta.append((group.category, a.source_id, b.source_id, text_a, text_b, expected))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, paper_a, paper_b, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=paper_a, paper_b=paper_b,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


@dataclass
class ThresholdSweepRow:
    threshold: float
    category: str
    total: int = 0
    correctly_flagged: int = 0
    incorrectly_flagged: int = 0
    missed: int = 0

    @property
    def false_positive_rate(self) -> float:
        """Fraction of "should NOT overlap" samples that wrongly cleared the
        threshold - the dangerous direction: a genuine gap getting demoted
        or a topically-unrelated pair getting flagged as addressing."""
        negatives = self.total - (self.correctly_flagged + self.missed)
        return self.incorrectly_flagged / negatives if negatives else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Fraction of "SHOULD overlap" samples that were missed - motivation/
        addressed language or a real benchmark-overlap that didn't clear
        the threshold and so failed to downgrade/flag anything."""
        positives = self.correctly_flagged + self.missed
        return self.missed / positives if positives else 0.0


def sweep_thresholds(samples: list[ThresholdSample], thresholds: list[float]) -> list[ThresholdSweepRow]:
    categories = sorted({s.category for s in samples})
    rows = []
    for threshold in thresholds:
        for category in categories:
            row = ThresholdSweepRow(threshold=threshold, category=category)
            for sample in samples:
                if sample.category != category:
                    continue
                row.total += 1
                cleared = sample.similarity >= threshold
                if sample.expected_high_overlap and cleared:
                    row.correctly_flagged += 1
                elif sample.expected_high_overlap and not cleared:
                    row.missed += 1
                elif not sample.expected_high_overlap and cleared:
                    row.incorrectly_flagged += 1
            rows.append(row)
    return rows
```

Add `PyYAML` usage note: check `pyproject.toml`/`requirements*.txt` for an existing `yaml` dependency before adding an import — `benchmark/store.py` already uses `yaml.safe_load`/`yaml.safe_dump`, so it's already a project dependency; no new dependency needed.

- [ ] **Step 4: Create the calibration groups skeleton**

Create `benchmark/gap_calibration_groups.yaml`:

```yaml
# Hand-curated groups for calibrating OWN_CONTRIBUTION_OVERLAP_THRESHOLD and
# ADDRESSING_SIMILARITY_THRESHOLD (see gaps/calibration.py and
# docs/superpowers/plans/2026-09-01-gap-evidence-grounding.md Task 14).
# These calibrate two DIFFERENT thresholds - keep the category you pick
# accurate to which one a group is meant to inform, since calibration.py
# routes each category to a different function.
#
# Each group is source_ids from benchmark/sample_manifest.csv (must already
# have a benchmark/annotations/arxiv_<source_id>.yaml file - reuses the
# existing Sec 25 annotations, no separate annotation format).
#
# category must be one of:
#
# --- calibrates OWN_CONTRIBUTION_OVERLAP_THRESHOLD (same-paper: does a
#     claim overlap with what its OWN paper contributed) ---
#   recurring_limitation_genuinely_unresolved  - 2+ papers independently
#     naming a similar still-open limitation, where EACH paper's own
#     contribution does NOT address it (expected_high_overlap=False per
#     member: that paper's own research_gap.remaining/limitations text
#     should not resemble that SAME paper's own main_contribution). Group
#     membership just helps you find a thematically real cluster; each
#     sample computed is still a same-paper self-pair.
#   recurring_limitation_topical_convergence   - 2+ papers in the same
#     narrow subfield each motivating their own contribution with similar
#     "prior work lacks X" phrasing, where X is what EACH one's own
#     contribution then provides (expected_high_overlap=True per member -
#     the FinRCA-Bench pattern). Same self-pair note as above.
#
# --- calibrates ADDRESSING_SIMILARITY_THRESHOLD (cross-paper: is Paper A's
#     gap similar enough to a DIFFERENT Paper B's contribution to justify a
#     reviewer warning - never proof B solved it) ---
#   high_value_addressing_match       - Paper B's contribution plausibly
#     bears on Paper A's stated gap; a warning here is worth showing a
#     reviewer (expected: should warn).
#   false_warning_topical_only        - Paper A and Paper B share only
#     domain/topical vocabulary with no real bearing on each other's gap;
#     a warning here would just be noise (expected: should NOT warn).
#   benchmark_evaluation_partial_solution - Paper B's contribution is a
#     benchmark, evaluation, dataset, or partial solution touching Paper
#     A's stated gap without solving the underlying problem (expected:
#     SHOULD still warn - that is the point of the signal - but this is
#     never read as "B solved it"; see golden test 7 in
#     tests/test_gaps_golden_regression.py for the runtime-behavior twin
#     of this category).
#
# Task 14 fills this in with real source_ids after reviewing actual corpus
# output - this skeleton intentionally ships with zero groups so
# cli_calibrate.py runs (and reports "no group-derived samples yet") rather
# than erroring on a missing file.
groups: []
```

- [ ] **Step 5: Write `cli_calibrate.py`**

```python
"""Runs gaps/calibration.py against the real corpus + real embedder,
mirroring extraction/cli_evaluate.py's structure: load ground truth, run
the measurement, print a table, write a JSON results file for the admin
dashboard / future re-runs.

Runs OWN_CONTRIBUTION_OVERLAP_THRESHOLD and ADDRESSING_SIMILARITY_THRESHOLD
as two separate sweeps over two separate sample pools, never combined -
they measure different things (own-paper overlap vs. cross-paper warning-
worthiness) and are calibrated independently (see calibration.py's module
docstring and Task 14).

Dry run, read-only: never writes to candidate_gaps or any pipeline table,
only to benchmark/gap_calibration_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from researchbridge.benchmark.cli_sample import DEFAULT_OUTPUT_DIR
from researchbridge.benchmark.store import load_all
from researchbridge.config import load_config
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.gaps.calibration import (
    addressing_signal_samples,
    load_calibration_groups,
    own_contribution_overlap_group_samples,
    own_contribution_overlap_samples,
    sweep_thresholds,
)

DEFAULT_GROUPS_PATH = Path("benchmark/gap_calibration_groups.yaml")
DEFAULT_RESULTS_PATH = Path("benchmark/gap_calibration_results.json")
DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Calibrate gap-signal overlap thresholds against real annotations.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--groups-file", type=Path, default=DEFAULT_GROUPS_PATH)
    parser.add_argument("--thresholds", type=str, default=None, help="comma-separated, e.g. 0.3,0.4,0.5")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    annotations = load_all(args.benchmark_dir)
    if not annotations:
        print(f"No annotations in {args.benchmark_dir}/annotations/ - run rb-benchmark-sample first.")
        return

    groups = load_calibration_groups(args.groups_file)
    thresholds = (
        [float(t) for t in args.thresholds.split(",")] if args.thresholds else DEFAULT_THRESHOLDS
    )

    embedder = SentenceTransformerEmbedder()

    own_overlap_samples = own_contribution_overlap_samples(annotations, embedder) + own_contribution_overlap_group_samples(
        annotations, groups, embedder
    )
    addressing_samples = addressing_signal_samples(annotations, groups, embedder)

    own_overlap_rows = sweep_thresholds(own_overlap_samples, thresholds) if own_overlap_samples else []
    addressing_rows = sweep_thresholds(addressing_samples, thresholds) if addressing_samples else []

    if not own_overlap_rows and not addressing_rows:
        print("No calibration samples available - check benchmark annotations and gap_calibration_groups.yaml.")
        return

    print("=== OWN_CONTRIBUTION_OVERLAP_THRESHOLD calibration (same-paper overlap) ===")
    if own_overlap_rows:
        _print_table(own_overlap_rows)
    else:
        print("(no samples - check gap_calibration_groups.yaml's recurring_limitation_* groups)")

    print("\n=== ADDRESSING_SIMILARITY_THRESHOLD calibration (cross-paper warning signal) ===")
    if addressing_rows:
        _print_table(addressing_rows)
    else:
        print("(no samples - check gap_calibration_groups.yaml's high_value_addressing_match/"
              "false_warning_topical_only/benchmark_evaluation_partial_solution groups)")

    _write_results_json(
        args.output_file,
        own_overlap_rows=own_overlap_rows,
        addressing_rows=addressing_rows,
        own_overlap_sample_count=len(own_overlap_samples),
        addressing_sample_count=len(addressing_samples),
    )
    print(f"\nWrote results to {args.output_file}")


def _print_table(rows) -> None:
    header = f"{'threshold':>10}{'category':<38}{'n':>6}{'false_pos_rate':>16}{'false_neg_rate':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.threshold:>10.2f}{row.category:<38}{row.total:>6}"
            f"{row.false_positive_rate:>16.3f}{row.false_negative_rate:>16.3f}"
        )


def _rows_to_json(rows) -> list[dict]:
    return [
        {
            "threshold": r.threshold,
            "category": r.category,
            "total": r.total,
            "false_positive_rate": r.false_positive_rate,
            "false_negative_rate": r.false_negative_rate,
        }
        for r in rows
    ]


def _write_results_json(
    path: Path, own_overlap_rows, addressing_rows, own_overlap_sample_count: int, addressing_sample_count: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "own_contribution_overlap": {
            "sample_count": own_overlap_sample_count,
            "rows": _rows_to_json(own_overlap_rows),
        },
        "addressing_similarity": {
            "sample_count": addressing_sample_count,
            "rows": _rows_to_json(addressing_rows),
        },
    }
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
```

Register a console-script entry point for this alongside the existing `rb-gaps-detect`/`rb-extraction-evaluate` entries — check `pyproject.toml`'s `[project.scripts]` section for the exact existing naming convention (likely `rb-<something>`) and add e.g. `rb-gaps-calibrate = "researchbridge.gaps.cli_calibrate:main"` matching it.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_gaps_calibration.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/gaps/calibration.py src/researchbridge/gaps/cli_calibrate.py benchmark/gap_calibration_groups.yaml tests/test_gaps_calibration.py pyproject.toml
git commit -m "feat(gaps): add threshold-calibration evaluation harness against hand-annotated benchmark"
```

---

## Task 14: Curate the cross-paper calibration groups, run the sweep, finalize the thresholds

**Files:**
- Modify: `benchmark/gap_calibration_groups.yaml`
- Modify: `src/researchbridge/gaps/signals.py` (only the two threshold constants' values and their comments)

**Interfaces:** none new — consumes everything built in Task 13.

This is the actual validation step: nothing here is unit-testable (it depends on real corpus content and a human judgment call), so it's kept separate from Task 13's TDD-able harness. The thresholds are only finalized after this task's Step 4 inspection — not before.

- [ ] **Step 1: Curate the groups, keeping the two thresholds' categories conceptually separate**

Using `benchmark/sample_manifest.csv`'s `domain` column to find same-subfield clusters, and reading the matching `benchmark/annotations/arxiv_<source_id>.yaml` files' `limitations`/`main_contribution`/`research_gap` text, identify real examples for each category `gap_calibration_groups.yaml`'s schema defines. Keep in mind while curating which threshold each category feeds — the two are calibrated independently in Step 3, and a mislabeled group would quietly cross-contaminate one threshold's evidence with the other's:

*For `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` (same-paper: does a claim overlap with what its own paper contributed):*
- **`recurring_limitation_genuinely_unresolved`**: 2-3 papers in the same domain whose `limitations`/`research_gap.remaining` text names a similar residual weakness that EACH paper's own `main_contribution` does not address (read both fields side by side, per paper — the annotation's own `main_contribution` text should read as unrelated to that same paper's limitation).
- **`recurring_limitation_topical_convergence`**: 2-3 papers in the same narrow subfield whose `limitations`/`research_gap.remaining` text motivates their own contribution in similar language, where EACH paper's own `main_contribution` does in fact cover what that same paper's limitation names (the FinRCA-Bench pattern, but drawn from real annotated papers rather than synthetic examples).

*For `ADDRESSING_SIMILARITY_THRESHOLD` (cross-paper: is Paper A's gap similar enough to a different Paper B's contribution to justify a reviewer warning — never proof B solved it):*
- **`high_value_addressing_match`**: a pair of papers A, B where B's `main_contribution` plausibly bears on A's stated `limitations`/`research_gap.remaining` — a real case where a reviewer would want to see the warning.
- **`false_warning_topical_only`**: a pair of papers A, B in the same broad domain (so they'd share some vocabulary) whose actual subject matter doesn't meaningfully relate — B's `main_contribution` should NOT read as bearing on A's gap despite domain overlap.
- **`benchmark_evaluation_partial_solution`**: a pair where B's `main_contribution` is a benchmark, evaluation, dataset, or partial solution touching A's stated gap, without B claiming to solve the underlying problem. This is the category golden test #7 already exercises at the runtime-behavior level — keep that test exactly as written (Task 10); this calibration category just makes sure the threshold that drives it is chosen deliberately rather than left at the `0.5` placeholder.

If the current 40-paper Sec 25 benchmark doesn't contain a clean real example for a category (plausible for `benchmark_evaluation_partial_solution` specifically, since it needs a benchmark-construction paper), it is acceptable to add 1-2 fresh annotations following the existing `benchmark/annotations/*.yaml` format/template (`benchmark/annotation_template.py`) for real arXiv papers already in the corpus that fit the pattern, rather than inventing synthetic text — every calibration sample must trace to a real paper, consistent with how the rest of this benchmark was built.

Fill in `benchmark/gap_calibration_groups.yaml`'s `groups: []` with the curated entries (source_ids + a one-line `note` explaining why each group was picked - this note is what a future recalibration reads to understand the labeling, exactly like `benchmark/annotations/*.yaml`'s own inline documentation).

- [ ] **Step 2: Run the sweep**

Run: `python -m researchbridge.gaps.cli_calibrate` (or `rb-gaps-calibrate` once the entry point from Task 13 Step 5 is installed)
Expected: two separate tables — one for `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` (`genuine_unresolved_gap`, `motivation_addressed`, `recurring_limitation_genuinely_unresolved`, `recurring_limitation_topical_convergence`) and one for `ADDRESSING_SIMILARITY_THRESHOLD` (`high_value_addressing_match`, `false_warning_topical_only`, `benchmark_evaluation_partial_solution`) — each at every threshold from 0.30 to 0.70. `benchmark/gap_calibration_results.json` is written with both sweeps kept in separate top-level keys.

- [ ] **Step 3: Apply the decision rule — separately for each threshold**

The two thresholds carry different failure costs and get different rules. Do not apply one rule to both.

**`OWN_CONTRIBUTION_OVERLAP_THRESHOLD`** (precision-first, unchanged from the original design): a false positive here (a genuine unresolved gap, or a genuinely-unresolved recurring limitation, wrongly registering as high-overlap) directly downgrades or excludes real evidence — this is the expensive direction. Scored against `genuine_unresolved_gap` + `recurring_limitation_genuinely_unresolved` (both "should NOT overlap") and `motivation_addressed` + `recurring_limitation_topical_convergence` (both "SHOULD overlap"). Pick the **smallest swept threshold** at which:
- both "should NOT overlap" categories' `false_positive_rate` is at its lowest observed value across the sweep, or within 0.05 of it (never trade away genuine-gap protection for a marginally better catch rate on the other two), **and**
- the "SHOULD overlap" categories' `false_negative_rate` is not the worst value observed across the sweep (don't pick a threshold so conservative it catches nothing).

If no swept value satisfies both, pick the threshold that minimizes `genuine_unresolved_gap`'s false-positive rate specifically, even at the cost of a higher false-negative rate elsewhere.

**`ADDRESSING_SIMILARITY_THRESHOLD`** (asymmetric — the signal is advisory, not gating, so its failure costs are different and the rule is *not* "catch as many warnings as possible"): a false positive here means `false_warning_topical_only` wrongly fires — a reviewer sees a warning note on a genuinely unrelated pairing. Repeated enough, that erodes trust in the signal until reviewers start ignoring it, which is worse for the system long-term than an occasional missed warning. A false negative here just means `high_value_addressing_match` or `benchmark_evaluation_partial_solution` misses a warning that would have been useful — the cluster's `gap_status` is unaffected either way (`find_addressing_papers` only ever adds a note and downgrades one level; it never invalidates). Per your explicit instruction: **do not optimize this threshold primarily to maximize detection of "should warn" examples.** Pick the **largest swept threshold** (i.e. the most conservative, least trigger-happy) at which `false_warning_topical_only`'s `false_positive_rate` is at or near zero (accept 0.0 if the sweep offers it; otherwise the lowest observed value), even if that means `high_value_addressing_match`/`benchmark_evaluation_partial_solution`'s `false_negative_rate` is high. Missing a weak addressing warning is the acceptable failure mode here; generating noise is not.

- [ ] **Step 4: Update the constants and their comments**

In `src/researchbridge/gaps/signals.py`, replace both constants' values and comments with the measured numbers, following `cluster.py`'s `DEFAULT_SIMILARITY_THRESHOLD` comment's exact prose pattern: state the measured per-category false-positive/false-negative rates at the chosen threshold (not just the final number), state what was tried and rejected (e.g. "0.5 let 2/7 genuine_unresolved_gap samples wrongly clear the threshold"), cite `benchmark/gap_calibration_results.json` as where the full sweep is recorded, and — for `ADDRESSING_SIMILARITY_THRESHOLD` specifically — restate in the comment that this constant is calibrated to minimize false warnings, not to maximize warning coverage, so a future reader doesn't "fix" it by lowering it to catch more `high_value_addressing_match` cases.

- [ ] **Step 5: Re-run the full gap test suite to confirm nothing broke**

Run: `pytest tests/test_gaps_signals.py tests/test_gaps_detect.py tests/test_gaps_persistence.py tests/test_gaps_batch.py tests/test_gaps_cli_detect.py tests/test_gaps_golden_regression.py tests/test_gaps_api.py tests/test_gaps_calibration.py -v`
Expected: PASS. The golden regression tests (Task 10) use fixed, hand-written overlap-triggering vocabulary deliberately far from any threshold boundary, so a reasonable adjustment within the swept range shouldn't flip any of them; if one does flip, move that scenario's wording further from the boundary rather than treating it as evidence the new threshold is wrong.

- [ ] **Step 6: Commit**

```bash
git add benchmark/gap_calibration_groups.yaml benchmark/gap_calibration_results.json src/researchbridge/gaps/signals.py
git commit -m "chore(gaps): finalize overlap thresholds from real-corpus calibration sweep"
```

---

## Self-Review Notes

**Spec coverage against the 9 numbered questions from the design conversation:**
1. Claims classified more reliably via surrounding context → Tasks 2-3 (`classify_claim`/`classify_cluster`), using `field_scope`/`self_resolution` as the "context" proxy available without re-parsing the paper.
2. `main_contribution`/`results` cross-check → Task 6 (`_own_contribution_overlaps`).
3. Cross-paper corroboration beyond similarity → Task 3's `independent_papers` requirement in `classify_cluster`.
4. Distinguishing "prior work has X; we address X" from "X remains unresolved" → Task 2's combined `self_resolution` + `own_contribution_overlap` scoring.
5. Evidence required before entering the review queue → Task 3's decision table + Task 7's `classify_cluster is None` drop.
6. Deterministic gap-strength status on the candidate object → Tasks 5/7/8 (`gap_status` persisted, `INSUFFICIENT_EVIDENCE`/`MOTIVATION`/`THEME_ONLY` never persisted, only logged).
7. Provenance (source paper, quote, claim_type, validation tier, signals, acceptance reason) → Task 11's `GapEvidenceOut`.
8. No LLM → every function in `gaps/signals.py` is regex/cosine only; confirmed in Global Constraints.
9. `DimensionCoverage` stays in-memory → no task touches `assessment/coverage.py`; called out explicitly in Global Constraints and the plan header.
10. No relevant papers vs. insufficient evidence vs. known/potential/strong → Task 7's `DetectionStatus` + `GapStatus`.

**Placeholder scan:** every step above has literal code, not a description of code; no "TODO"/"handle edge cases" language appears outside of the two explicitly-flagged, genuinely-deferred calibration constants (Tasks 13-14 exist specifically to resolve those with a real evaluation harness and a stated decision rule — not just documentation of the `0.5` starting values — consistent with how `cluster.py`/`semantic.py` document their own calibration history).

**Type consistency:** `ClaimClassification`/`ClassifiedMember`/`AddressingMatch`/`GapStatus`/`ClaimRole` are defined once in Task 2-4 and reused with identical names/shapes in Tasks 6-8, 11-12. `CandidateGapDraft`'s three new fields (`gap_status`, `resolution_note`, `evidence_roles`) are introduced in Task 7 and consumed with matching names in Task 8. `DetectionStatus`/`GapDetectionResult` introduced in Task 7 are consumed identically in Task 9. `ThresholdSample`/`ThresholdSweepRow`/`CalibrationGroup` are defined once in Task 13 and consumed identically by Task 14's CLI run.

**Calibration methodology coverage against the 6 required categories, and which threshold each informs:**
1. Genuine unresolved gaps → `genuine_unresolved_gap` samples, drawn directly from every annotated paper's real `research_gap.remaining` vs its own `main_contribution` (Task 13, `own_contribution_overlap_samples`) → calibrates `OWN_CONTRIBUTION_OVERLAP_THRESHOLD`.
2. Motivation statements → `motivation_addressed` samples, from `research_gap.addressed` vs its own `main_contribution` (Task 13, same function) → calibrates `OWN_CONTRIBUTION_OVERLAP_THRESHOLD`.
3. Contribution statements → covered two ways: as the positive half of `motivation_addressed` above, and independently regression-tested at the extraction-validation layer by Task 10's golden test 3 (a contribution-flavored sentence never even reaches `ExtractedClaim` typed `research_gap`).
4. Benchmark/evaluation papers that overlap strongly but don't solve → `benchmark_evaluation_partial_solution`, a curated cross-paper group scored by `addressing_signal_samples` (Task 14, Step 1) → calibrates `ADDRESSING_SIMILARITY_THRESHOLD`, with an explicit non-goal (do not optimize to catch more of these — see Task 14 Step 3's asymmetric decision rule) and a runtime-behavior twin in Task 10's golden test 7, kept unchanged.
5. Recurring limitations genuinely unresolved → `recurring_limitation_genuinely_unresolved`, a curated group of same-paper self-pairs (Task 14, Step 1) → calibrates `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` (this is a same-paper signal — each member's own limitation against that same paper's own contribution — even though a thematic group is what's curated to find real examples).
6. Recurring limitations that are merely topical convergence → `recurring_limitation_topical_convergence`, same same-paper-self-pair mechanism (Task 14, Step 1) → calibrates `OWN_CONTRIBUTION_OVERLAP_THRESHOLD`.

The two additional addressing-signal categories (`high_value_addressing_match`, `false_warning_topical_only`) exist because `ADDRESSING_SIMILARITY_THRESHOLD` needs its own positive and negative controls distinct from categories 1/2/5/6 above — none of which are cross-paper comparisons, so none of them can calibrate a cross-paper signal.

Every category traces to real paper text (existing Sec 25 annotations, or 1-2 freshly annotated real papers if the existing 40 lack a clean example — never synthetic/invented text). `OWN_CONTRIBUTION_OVERLAP_THRESHOLD` and `ADDRESSING_SIMILARITY_THRESHOLD` are swept and finalized as two independent evaluations (Task 14 Step 2's two separate tables), each with its own decision rule (Step 3) — precision-first/symmetric for the own-overlap threshold, asymmetric and warning-noise-averse for the addressing threshold — and neither is touched before Step 2's real-corpus results exist to inspect.
