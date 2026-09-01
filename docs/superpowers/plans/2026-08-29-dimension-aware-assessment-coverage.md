# Dimension-Aware Evidence Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a deterministic, in-memory "dimension coverage" layer between retrieval and the existing `assess_*` modules so the assessment report can show *why* it reached its recommendation (per-dimension evidence coverage), fix the confirmed claim-misclassification and evidence-duplication bugs, and distinguish "insufficient evidence" from "checked, found nothing" everywhere the report currently collapses both to the same sentence.

**Architecture:**
```
ResearchInput
     v
Retrieval (unchanged: search_by_text, top_k)
     v
extract_dimensions(query_text)              <- NEW, deterministic RAKE-style keyword
     v                                          extraction, no new dependency
compute_dimension_coverage(dimensions,
     papers_with_claims)                     <- NEW, pure function, reuses claims
     v                                          build.py already fetches
assess_novelty / assess_gap / assess_applications /
assess_feasibility / assess_risks / assess_recommendation  <- existing modules,
     v                                          each gains an optional coverage input
ResearchAssessment (unchanged schema)
     v
Explainable report (existing TEXT/String/JSONB columns,
     reused more fully, not restructured)
```
`DimensionCoverage` is a plain dataclass computed fresh inside `build_assessment()` on every run. It is never persisted as its own table or column — see "Migration implications" below for exactly how each downstream signal survives without new schema.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x ORM, pytest, existing `Embedder` protocol (`embed_texts`), `sklearn.cluster.AgglomerativeClustering` (already a dependency, used by `gaps/cluster.py`). No new third-party dependency.

**Spec:** This plan implements the architecture proposed in conversation (no separate spec doc); the governing constraints are reproduced in "Global Constraints" below, copied verbatim from the user's approval message.

## Global Constraints

- `DimensionCoverage` is an in-memory/domain representation produced during `build_assessment`, **not** a new DB table. `ResearchAssessment` stays the primary persisted product object.
- Dimension extraction must be deterministic, explainable, and lightweight — no generative LLM, no new heavyweight NLP dependency (no spaCy/nltk). Define exactly what an "idea dimension" is before writing extraction code (Task 1 does this).
- Evidence grounding is a hard invariant: every `evidence_id` anywhere in `DimensionCoverage` must trace back to a real, already-persisted `Evidence`/`ExtractedClaim` row. The coverage layer must never create new `Evidence` rows or invent quotes.
- `partially_addressed` / `not_found` are evidence-coverage observations, never presented as author-stated research gaps.
- Novelty must not collapse to `not_found dimension -> novel`. Novelty stays a corpus-similarity signal, never proof of global originality, and must define what happens when corpus evidence is insufficient.
- The recommendation chain must be inspectable: novelty signal, research-gap evidence, technical grounding, and per-dimension coverage must all be visible in the persisted/report-facing output, not discarded after `build_assessment` returns.
- `not_found` (evidence was sufficient to investigate, nothing supported it) must never be collapsed into the same sentence as `not_assessed` / insufficient evidence (evidence was insufficient to investigate at all).
- Applications stay strictly extractive — the coverage layer must never generate product/technology opportunities or applications; it may only let the report distinguish "no relevant papers" from "relevant papers, no explicit application claim."
- Risks stay strictly extractive — a coverage gap ("no evidence found for concept drift") must render as an evidence limitation, never auto-promoted into a risk statement.
- Preserve: PostgreSQL + pgvector, current retrieval (`search_by_text`, `top_k`), current extraction pipeline, current `Evidence`/`ExtractedClaim` model, free/open-source-first, deterministic/non-generative assessment architecture, the `ResearchInput -> ResearchAssessment` contract.
- Minimal migration footprint: this plan adds **zero** new tables and **zero** new columns (see "Migration implications" below for how every new signal is carried inside existing column value-spaces).

## Migration implications: none

Every new signal is carried without altering the schema, by using existing columns more fully than they are today:

| New signal | Where it lives | Why no migration is needed |
|---|---|---|
| Per-dimension coverage table (`federated learning -> established`, ...) | Appended as a formatted block at the end of `ResearchAssessment.novelty_reasoning` (existing `Text`, nullable) | `novelty_reasoning` is already free text produced by `assess_novelty`; it just gains a second, clearly-labeled paragraph. |
| Risk-relevant coverage gaps (dimensions with no supporting literature) | Appended as a separate, clearly-labeled block at the end of `ResearchAssessment.risks_and_limitations` (existing `Text`, nullable) | Same reasoning; kept structurally separate from the actual risk lines so it can never be misread as a stated risk. |
| Research-gap `not_assessed` vs `not_found` vs `found` | `ResearchAssessment.research_gap_source` (existing `String`, nullable) gains two new string values (`"no_relevant_evidence"`, `"checked_no_gap_found"`) alongside the existing `"reused_candidate_gap"` / `"input_specific"` | `String` columns don't require a migration to accept new values — only the Python-level vocabulary changes. |
| Applications `not_assessed` vs `no_evidence` vs `found` | `ResearchAssessment.potential_applications` (existing `JSONB`, nullable) now stores `None` for not_assessed and `[]` (empty list) for no_evidence, instead of collapsing both to `None` | `list[dict] \| None` already permits an empty list; today's code just never produces one. |
| Recommendation chain narrative (novelty/gap/feasibility/dimension summary -> recommendation) | Computed **at read time** in a new pure function, from fields already listed above — never persisted | Fully derivable from already-persisted columns, so nothing needs to survive between the write (`build_assessment`) and the read (API/export) other than what's already there. |

If, after this ships, the coverage table proves useful as a queryable/filterable structure in its own right (e.g. "show me all assessments where `concept drift` is `not_found`"), that's the trigger for a real follow-up migration — deliberately deferred, per the constraint above.

---

## What "idea dimension" means here (read before Task 1)

An **idea dimension** is a short noun phrase drawn **verbatim** from the `ResearchInput`'s own query text (the same `query_text` `build_assessment` already computes — raw idea text, or `build_research_representation()`'s output for uploaded documents). It is not a category from a fixed taxonomy and not something invented or paraphrased — it is literally a substring of the input, the same "verbatim grounding" discipline the rest of this codebase already applies to extracted claims.

**Mechanism: RAKE (Rapid Automatic Keyword Extraction, Rose et al. 2010).** Chosen over the alternatives considered:

- *Reusing `HybridExtractor`* (rejected, per the constraint) — it's trained/tuned to classify a *paper's* sentences into `problem`/`method`/`dataset`/etc. fields; an idea's raw text is a different shape of input (often one sentence, no abstract structure), and forcing it through that pipeline would silently repurpose a tool built for something else.
- *Free-form n-gram + embedding-cluster dedup* (considered) — more general, but non-deterministic-feeling (cluster boundaries shift with embedder/threshold) and harder to explain to a reader ("why these five phrases and not those five?").
- *Fixed keyword vocabulary lookup* (considered) — explainable and deterministic, but doesn't generalize past the seed list; every new domain requires editing a Python dict, which is the "unrestricted NLP research project" trap in reverse (a maintenance burden with a different shape).
- **RAKE** (chosen) — deterministic, well-understood, off-the-shelf (no tuning against this specific example required to justify it), and requires nothing beyond a stopword list. It splits text into candidate phrases at stopword/punctuation boundaries, then scores each candidate by word co-occurrence degree — phrases built from words that co-occur with many other content words outrank single common words. It is naturally explainable: "this phrase was picked because it's a maximal run of content words in your own sentence, ranked by how often its words co-occur with other content words."

Worked example (used verbatim in Task 1's tests and Task 11's acceptance test):

> "Can we build a privacy-preserving federated learning system for detecting financial fraud across multiple banks without sharing raw transaction data, while remaining robust to concept drift and highly imbalanced fraud classes?"

Stopword-boundary splitting yields candidates such as: `build privacy-preserving federated learning system`, `detecting financial fraud`, `multiple banks`, `sharing raw transaction data`, `remaining robust`, `concept drift`, `highly imbalanced fraud classes`. These are not perfectly clean (`federated learning` sits inside a longer phrase rather than standing alone) — that's acceptable: dimension *matching* against paper claims in Task 2 uses embedding similarity, not exact substring match, so a longer phrase containing "federated learning" still matches papers about federated learning. Task 1's calibration step (folded into Task 12) is where phrase quality gets a real spot-check against the live corpus, the same way `novelty.py`'s and `feasibility.py`'s thresholds were originally calibrated.

---

## Task 1: Deterministic dimension extraction

**Files:**
- Create: `src/researchbridge/assessment/dimensions.py`
- Test: `tests/test_assessment_dimensions.py`

**Interfaces:**
- Produces: `IdeaDimension` (dataclass: `label: str`), `extract_dimensions(text: str, max_dimensions: int = 8) -> list[IdeaDimension]`. No embedder needed for this step — pure string processing.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from researchbridge.assessment.dimensions import extract_dimensions

FRAUD_IDEA = (
    "Can we build a privacy-preserving federated learning system for detecting "
    "financial fraud across multiple banks without sharing raw transaction data, "
    "while remaining robust to concept drift and highly imbalanced fraud classes?"
)


def test_empty_text_yields_no_dimensions() -> None:
    assert extract_dimensions("") == []
    assert extract_dimensions("   ") == []


def test_extracts_multiple_dimensions_from_the_fraud_idea() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)

    labels = [d.label.lower() for d in dimensions]
    assert len(dimensions) >= 3
    assert any("concept drift" in label for label in labels)
    assert any("fraud" in label for label in labels)


def test_dimension_labels_are_verbatim_substrings_of_the_input() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)

    lowered_source = FRAUD_IDEA.lower()
    for dimension in dimensions:
        assert dimension.label.lower() in lowered_source


def test_respects_max_dimensions_cap() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA, max_dimensions=2)
    assert len(dimensions) <= 2


def test_no_duplicate_or_substring_overlapping_dimensions() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)
    labels = [d.label.lower() for d in dimensions]

    assert len(labels) == len(set(labels))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                assert a not in b, f"{a!r} is a substring of {b!r}; should have been deduped"


def test_single_short_sentence_still_yields_something() -> None:
    dimensions = extract_dimensions("A lock-free queue for concurrent systems.")
    assert len(dimensions) >= 1
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_dimensions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.assessment.dimensions'`

- [x] **Step 3: Implement `dimensions.py`**

```python
"""Deterministic idea-dimension extraction (RAKE-style keyword extraction).

An "idea dimension" is a short noun phrase drawn VERBATIM from the input
text - never invented, paraphrased, or looked up in a fixed vocabulary.
See docs/superpowers/plans/2026-08-29-dimension-aware-assessment-coverage.md
("What 'idea dimension' means here") for why RAKE was chosen over reusing
HybridExtractor, free-form n-gram+embedding clustering, or a fixed keyword
list: it's deterministic, explainable without per-domain tuning, and needs
no new dependency.

Standard RAKE (Rose et al. 2010): split text into candidate phrases at
stopword/punctuation boundaries, score each word by degree(word)/
frequency(word) (co-occurrence with other content words vs. how often it
appears alone), score each phrase as the sum of its words' scores, rank
phrases by score, and drop phrases that are pure substrings of an
already-kept higher-scoring phrase (so "fraud" doesn't survive alongside
"financial fraud").

Deliberately NOT tuned against any specific idea's expected output - the
stopword list is a generic, mid-sized English function-word list, not
hand-picked for the fraud/federated-learning worked example. See Task 12
for the live-corpus calibration pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PHRASE_WORDS = 5

_STOPWORDS = frozenset(
    """
    a an the and or but if then than so because of to in on for with without
    by at from as is are was were be been being this that these those it
    its we our you your they their he she his her can could should would
    will shall may might must do does did not no nor while across using use
    via into onto up down over under between among each all any some such
    which what who whom whose when where why how also yet both either
    neither more most much many several own same other only just about
    again further once here there all any both each few more most other
    some such nor too very s t don now build remain remaining provide
    providing based upon
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


@dataclass
class IdeaDimension:
    label: str


def extract_dimensions(text: str, max_dimensions: int = 8) -> list[IdeaDimension]:
    if not text or not text.strip():
        return []

    tokens = _TOKEN_RE.findall(text)
    candidates = _candidate_phrases(tokens)
    if not candidates:
        return []

    word_score = _score_words(candidates)
    scored = [(phrase, sum(word_score[w.lower()] for w in phrase.split())) for phrase in candidates]

    # stable order: highest score first, ties broken by first appearance
    order_index = {phrase: i for i, phrase in enumerate(candidates)}
    scored.sort(key=lambda item: (-item[1], order_index[item[0]]))

    selected: list[str] = []
    for phrase, _score in scored:
        lowered = phrase.lower()
        if any(lowered in kept.lower() or kept.lower() in lowered for kept in selected):
            continue
        selected.append(phrase)
        if len(selected) >= max_dimensions:
            break

    # re-order by first appearance in the text for a readable, stable report
    selected.sort(key=lambda phrase: order_index[phrase])
    return [IdeaDimension(label=phrase) for phrase in selected]


def _candidate_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token.lower() in _STOPWORDS:
            if current:
                phrases.extend(_split_to_max_length(current))
                current = []
            continue
        current.append(token)
    if current:
        phrases.extend(_split_to_max_length(current))
    # de-duplicate identical phrases (case-insensitive), keep first occurrence
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(phrase)
    return unique


def _split_to_max_length(run: list[str]) -> list[str]:
    """A stopword-bounded run longer than MAX_PHRASE_WORDS is still one
    candidate phrase (RAKE doesn't cap phrase length) - but very long runs
    (rare; usually a sentence with no stopwords at all) are chopped into
    fixed windows so one candidate can't swallow half the input."""
    if len(run) <= MAX_PHRASE_WORDS:
        return [" ".join(run)]
    return [" ".join(run[i : i + MAX_PHRASE_WORDS]) for i in range(0, len(run), MAX_PHRASE_WORDS)]


def _score_words(candidates: list[str]) -> dict[str, float]:
    frequency: dict[str, int] = {}
    degree: dict[str, int] = {}
    for phrase in candidates:
        words = phrase.lower().split()
        length = len(words)
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
            degree[word] = degree.get(word, 0) + length
    return {word: degree[word] / frequency[word] for word in frequency}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_dimensions.py -v`
Expected: PASS. If `test_extracts_multiple_dimensions_from_the_fraud_idea` or the substring-dedup test fails on the exact worked example, adjust `_STOPWORDS`/`MAX_PHRASE_WORDS` (not the test) until it passes — the test encodes the actual requirement, not an implementation detail.

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/dimensions.py tests/test_assessment_dimensions.py
git commit -m "feat(assessment): add deterministic RAKE-based dimension extraction"
```

---

## Task 2: Dimension coverage computation

**Files:**
- Create: `src/researchbridge/assessment/coverage.py`
- Test: `tests/test_assessment_coverage.py`

**Interfaces:**
- Consumes: `IdeaDimension` (Task 1), `Embedder.embed_texts(texts: list[str]) -> list[list[float]]` (existing protocol, `src/researchbridge/embedding/base.py`), the same `EvidencedPaper = tuple[str, float, list[ClaimRecord]]` shape `novelty.py` already defines (`ClaimRecord = tuple[str, str, uuid.UUID]` = `(claim_type, text, evidence_id)`).
- Produces: `DimensionCoverage` (dataclass: `dimension: str`, `status: str` in `{"established", "partially_addressed", "weak_evidence", "not_found", "not_assessed"}`, `evidence_ids: list[uuid.UUID]`, `supporting_paper_titles: list[str]`), `compute_dimension_coverage(dimensions, papers_by_distance, embedder, relevance_distance=FAR_DISTANCE) -> list[DimensionCoverage]`.

This module never touches the database and never creates `Evidence` rows — it only reads the `claims` lists already fetched by `build.py`'s `_claims_for_paper`, so the hard evidence-grounding invariant holds by construction: every `evidence_id` it returns came from an existing `ExtractedClaim`/`Evidence` row.

- [x] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.coverage import compute_dimension_coverage
from researchbridge.assessment.dimensions import IdeaDimension

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake used by tests/test_assessment_gap.py: cosine similarity of
    two texts is exactly their fraction of shared distinct words, so test
    expectations are easy to reason about by hand."""

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


NEAR = 0.1
FAR = 0.9


def test_not_assessed_when_no_relevant_papers_retrieved(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Off-topic Paper", FAR, [_claim("method", "concept drift concept drift concept drift")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert len(result) == 1
    assert result[0].status == "not_assessed"
    assert result[0].evidence_ids == []


def test_not_found_when_relevant_papers_exist_but_none_mention_the_dimension(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Relevant Paper", NEAR, [_claim("method", "a completely unrelated technique")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "not_found"
    assert result[0].evidence_ids == []


def test_weak_evidence_when_exactly_one_relevant_paper_matches(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("limitations", "concept drift concept drift remains unaddressed")]),
        ("Paper B", NEAR, [_claim("method", "an unrelated method entirely")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "weak_evidence"
    assert len(result[0].supporting_paper_titles) == 1
    assert result[0].supporting_paper_titles == ["Paper A"]


def test_partially_addressed_when_multiple_papers_only_flag_it_as_a_limitation(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("limitations", "concept drift concept drift is unhandled")]),
        ("Paper B", NEAR, [_claim("research_gap", "concept drift concept drift remains open")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "partially_addressed"
    assert len(result[0].supporting_paper_titles) == 2


def test_established_when_multiple_papers_affirmatively_cover_it(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("method", "handles concept drift concept drift directly")]),
        ("Paper B", NEAR, [_claim("results", "concept drift concept drift adaptation improved accuracy")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "established"
    assert len(result[0].supporting_paper_titles) == 2


def test_evidence_ids_always_trace_back_to_input_claims(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    method_claim = _claim("method", "handles concept drift concept drift directly")
    other_claim = _claim("method", "handles concept drift concept drift directly")
    papers = [("Paper A", NEAR, [method_claim]), ("Paper B", NEAR, [other_claim])]

    result = compute_dimension_coverage(dims, papers, embedder)

    all_input_evidence_ids = {method_claim[2], other_claim[2]}
    assert set(result[0].evidence_ids) <= all_input_evidence_ids


def test_multiple_dimensions_are_each_scored_independently(embedder) -> None:
    dims = [IdeaDimension(label="concept drift"), IdeaDimension(label="class imbalance")]
    papers = [
        ("Paper A", NEAR, [_claim("method", "handles concept drift concept drift directly")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    by_label = {r.dimension: r for r in result}
    assert by_label["concept drift"].status == "weak_evidence"
    assert by_label["class imbalance"].status == "not_found"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.assessment.coverage'`

- [x] **Step 3: Implement `coverage.py`**

```python
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
# see Task 12 for the live-corpus calibration pass. Starting point matches
# the ballpark semantic.py found workable for sentence-vs-anchor matching
# (MEDIUM_CONFIDENCE_SIMILARITY = 0.28).
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
    relevant = [p for p in papers_by_distance if p[1] <= relevance_distance and p[2]]
    if not dimensions:
        return []
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


def _status_for(label: str, matches: list[tuple[str, "ClaimRecord"]]) -> DimensionCoverage:
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

    has_affirmative = any(claim[0] in _AFFIRMATIVE_CLAIM_TYPES for claims in papers_by_title.values() for claim in claims)
    status = "established" if has_affirmative else "partially_addressed"
    return DimensionCoverage(
        dimension=label, status=status, evidence_ids=evidence_ids, supporting_paper_titles=titles
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_coverage.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/coverage.py tests/test_assessment_coverage.py
git commit -m "feat(assessment): add pure-function per-dimension evidence coverage"
```

---

## Task 3: Fix the `main_contribution` -> "Problems already addressed" misclassification

**Files:**
- Modify: `src/researchbridge/assessment/existing_solutions.py:41`
- Test: `tests/test_assessment_existing_solutions.py`

**Interfaces:** No signature changes — `build_existing_solutions()` keeps its exact current shape.

- [x] **Step 1: Write the failing regression test**

Add to `tests/test_assessment_existing_solutions.py`:

```python
def test_main_contribution_claims_are_not_classified_as_problems() -> None:
    # a paper's own contribution ("we show that X improves Y") describes
    # what the paper CONTRIBUTES, not a problem it addresses - putting it
    # under "Problems already addressed" was a section-mapping bug
    papers = [("Paper A", [("main_contribution", "We show that our method improves accuracy by 12%.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None or "Problems already addressed" not in result.text
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assessment_existing_solutions.py::test_main_contribution_claims_are_not_classified_as_problems -v`
Expected: FAIL — `main_contribution` currently lands under "Problems already addressed" (`existing_solutions.py:41`)

- [x] **Step 3: Fix the section mapping**

In `src/researchbridge/assessment/existing_solutions.py`, change:

```python
    ("problems_addressed", "Problems already addressed", ("problem", "main_contribution")),
```

to:

```python
    ("problems_addressed", "Problems already addressed", ("problem",)),
```

Also tighten the module docstring's claim-type list note (it currently doesn't call out `main_contribution` specifically, so no further doc change is required) and update the inline comment at the top of `_SECTIONS` if it references `main_contribution` — it doesn't today, so no change needed there.

- [x] **Step 4: Run full existing_solutions test file to verify pass and no regressions**

Run: `pytest tests/test_assessment_existing_solutions.py -v`
Expected: PASS, all tests including the new one and the pre-existing `test_groups_claims_by_question_not_by_paper` (which will need updating — see Step 4a).

- [x] **Step 4a: Update the now-incorrect existing test**

`test_groups_claims_by_question_not_by_paper` doesn't use `main_contribution`, so it's unaffected. Confirm via the run in Step 4; if any other existing test in the file asserts `main_contribution` appears under "Problems already addressed", update it to expect it appears nowhere (since no section claims it anymore) — grep first:

```bash
grep -n "main_contribution" tests/test_assessment_existing_solutions.py
```

If that grep returns only the new test from Step 1, no further changes are needed.

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/existing_solutions.py tests/test_assessment_existing_solutions.py
git commit -m "fix(assessment): stop classifying main_contribution claims as problems"
```

---

## Task 4: Dimension-aware novelty

**Files:**
- Modify: `src/researchbridge/assessment/novelty.py`
- Test: `tests/test_assessment_novelty.py`

**Interfaces:**
- Consumes: `DimensionCoverage` (Task 2).
- Produces: `NoveltyResult` gains `dimension_coverage: list[DimensionCoverage] = field(default_factory=list)`. `assess_novelty()` gains an optional `dimension_coverages: list[DimensionCoverage] | None = None` parameter — when `None` or empty, behavior is byte-for-byte identical to today (nearest-distance-only), so every existing caller and every existing test in this file keeps passing unmodified. This is a strict additive superset, not a breaking change.

**The aggregate rule** (only used when `dimension_coverages` is non-empty): score dimensions excluding any `not_assessed` ones. Let `n` = count of scored dimensions, `covered` = count with status in `{established, partially_addressed}`, `established` = count with status `established`.

- `covered / n >= 0.7` and `established / n >= 0.5` -> `"low"` (most dimensions of the idea are individually well-represented in the retrieved literature, even if no single paper is a close overall match)
- `covered / n <= 0.3` -> `"high"` (most dimensions have little to no supporting evidence in this corpus sample — hedged exactly like today's high-novelty language, never "this idea is novel")
- otherwise -> `"medium"`
- if every dimension is `not_assessed` (i.e. `n == 0`), fall back to the existing nearest-distance-only logic unchanged — insufficient dimension-level evidence doesn't mean insufficient novelty evidence; the existing single-paper signal is still meaningful and shouldn't be silently discarded.

The `reasoning` text keeps its existing nearest-distance sentence (unchanged, still the primary explanation) and, when dimension coverage was used, appends a second paragraph: a literal `"Dimension coverage:\n  <label> -> <status>\n  ..."` block — this is the exact block persisted into `novelty_reasoning`, per the "Migration implications" table.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_assessment_novelty.py`:

```python
from researchbridge.assessment.coverage import DimensionCoverage


def test_dimension_coverage_absent_keeps_legacy_nearest_distance_behavior() -> None:
    # no dimension_coverages passed at all - identical to every pre-existing
    # test in this file, confirming the parameter is truly additive
    result = assess_novelty([("Existing Solution Paper", 0.1, _claims("tested only offline"))])
    assert result.level == "low"
    assert result.dimension_coverage == []


def test_falls_back_to_nearest_distance_when_all_dimensions_not_assessed() -> None:
    coverages = [DimensionCoverage(dimension="concept drift", status="not_assessed")]
    result = assess_novelty(
        [("Existing Solution Paper", 0.1, _claims("tested only offline"))], dimension_coverages=coverages
    )
    assert result.level == "low"  # legacy nearest-distance rule, unaffected by not_assessed-only coverage


def test_low_novelty_when_most_dimensions_are_established() -> None:
    coverages = [
        DimensionCoverage(dimension="federated learning", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="privacy preservation", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="fraud detection", status="established", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty(
        [("Somewhat Related Paper", 0.5, _claims("a moderately related claim"))], dimension_coverages=coverages
    )
    assert result.level == "low"


def test_high_novelty_when_most_dimensions_are_not_found() -> None:
    coverages = [
        DimensionCoverage(dimension="concept drift", status="not_found"),
        DimensionCoverage(dimension="class imbalance", status="not_found"),
        DimensionCoverage(dimension="cross-bank setting", status="weak_evidence", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty(
        [("Somewhat Related Paper", 0.5, _claims("a moderately related claim"))], dimension_coverages=coverages
    )
    assert result.level == "high"


def test_high_novelty_reasoning_still_never_claims_absolute_novelty_with_coverage() -> None:
    coverages = [DimensionCoverage(dimension="concept drift", status="not_found")]
    result = assess_novelty(
        [("Distant Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages
    )
    lowered = result.reasoning.lower()
    assert "this idea is novel" not in lowered
    assert "genuinely novel" not in lowered


def test_reasoning_includes_a_dimension_coverage_block() -> None:
    coverages = [
        DimensionCoverage(dimension="concept drift", status="not_found"),
        DimensionCoverage(dimension="federated learning", status="established", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)

    assert "Dimension coverage:" in result.reasoning
    assert "concept drift -> not_found" in result.reasoning
    assert "federated learning -> established" in result.reasoning


def test_medium_novelty_when_coverage_is_mixed() -> None:
    coverages = [
        DimensionCoverage(dimension="a", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="b", status="not_found"),
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "medium"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_novelty.py -v`
Expected: FAIL — `assess_novelty()` doesn't accept `dimension_coverages` yet, `NoveltyResult` has no `dimension_coverage` field.

- [x] **Step 3: Implement the change**

In `src/researchbridge/assessment/novelty.py`, add the import and update the dataclass and function:

```python
from researchbridge.assessment.coverage import DimensionCoverage

...

@dataclass
class NoveltyResult:
    level: str  # high | medium | low | not_assessed
    reasoning: str
    evidence_ids: list[uuid.UUID]
    dimension_coverage: list[DimensionCoverage] = field(default_factory=list)


_COVERED_STATUSES = frozenset({"established", "partially_addressed"})


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

    return NoveltyResult(level=level, reasoning=reasoning, evidence_ids=evidence_ids, dimension_coverage=dimension_coverages or [])


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
```

Note: `field` must already be imported (`from dataclasses import dataclass, field`) — add `field` to the existing `from dataclasses import dataclass` line at the top of the file.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_novelty.py -v`
Expected: PASS — both the new dimension-aware tests and every pre-existing nearest-distance-only test (unaffected, since `dimension_coverages` defaults to `None`).

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/novelty.py tests/test_assessment_novelty.py
git commit -m "feat(assessment): make novelty an aggregate multi-dimension signal, not nearest-paper-only"
```

---

## Task 5: Distinguish research-gap `not_assessed` from `not_found`

**Files:**
- Modify: `src/researchbridge/assessment/gap.py`
- Test: `tests/test_assessment_gap.py`

**Interfaces:** `GapAssessmentResult` gains `status: str` (`"not_assessed" | "not_found" | "found"`). `source` keeps its exact current meaning and values (`"reused_candidate_gap" | "input_specific" | None`) — `status` is a new, additional field, not a replacement.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_assessment_gap.py`:

```python
def test_status_is_not_assessed_when_no_relevant_papers_retrieved(session_factory, embedder) -> None:
    session = session_factory()
    result = assess_research_gap(session, [], embedder)
    session.close()
    assert result.status == "not_assessed"
    assert result.text is None


def test_status_is_not_found_when_relevant_papers_exist_but_nothing_matches(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "an unrelated method")  # relevant paper, no gap-signaling claim
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert result.status == "not_found"
    assert result.text is None


def test_status_is_found_when_an_explicit_gap_claim_exists(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert result.status == "found"
    assert result.text is not None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_gap.py -v`
Expected: FAIL — `GapAssessmentResult` has no `status` attribute.

- [x] **Step 3: Implement the change**

In `src/researchbridge/assessment/gap.py`:

```python
@dataclass
class GapAssessmentResult:
    source: str | None  # "reused_candidate_gap" | "input_specific" | None
    text: str | None
    candidate_gap_id: uuid.UUID | None
    evidence_ids: list[uuid.UUID]
    is_closely_grounded: bool = False
    status: str = "not_assessed"  # "not_assessed" | "not_found" | "found"


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
```

Replace every remaining reference to the old `_NONE_RESULT` name (there should be none left after this edit — grep to confirm):

```bash
grep -n "_NONE_RESULT" src/researchbridge/assessment/gap.py
```

Expected: no matches after the edit.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_gap.py -v`
Expected: PASS — including every pre-existing test in the file (they only assert on `source`/`text`/`candidate_gap_id`/`evidence_ids`/`is_closely_grounded`, none of which changed shape).

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/gap.py tests/test_assessment_gap.py
git commit -m "feat(assessment): distinguish gap not_assessed from not_found"
```

---

## Task 6: Distinguish applications `not_assessed` from `no_evidence`

**Files:**
- Modify: `src/researchbridge/assessment/applications.py`
- Modify: `src/researchbridge/assessment/build.py` (the `potential_applications` assembly at lines 90-101)
- Test: `tests/test_assessment_applications.py`
- Test: `tests/test_assessment_build.py`

**Interfaces:** `ApplicationsResult` gains `status: str` (`"not_assessed" | "no_evidence" | "found"`). `build.py` persists `potential_applications` as `None` for `not_assessed`, `[]` (empty list) for `no_evidence`, and the existing populated-list shape for `found` — no schema change, since the column is already `list | None`.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_assessment_applications.py` (create the file with these tests plus imports if the module-level fixtures aren't already present — check the existing file first and reuse its `_paper`/`_claim`/`embedder`-equivalent helpers rather than duplicating them):

```python
def test_status_not_assessed_when_no_relevant_papers(session_factory) -> None:
    session = session_factory()
    result = assess_applications(session, [])
    session.close()
    assert result.status == "not_assessed"
    assert result.applications is None


def test_status_no_evidence_when_relevant_papers_have_no_application_claim(session_factory) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "method", "an unrelated method")
    session.commit()

    result = assess_applications(session, [(paper.id, NEAR)])

    session.close()
    assert result.status == "no_evidence"
    assert result.applications == []


def test_status_found_when_an_application_claim_exists(session_factory) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "applications", "used in real-time fraud monitoring")
    session.commit()

    result = assess_applications(session, [(paper.id, NEAR)])

    session.close()
    assert result.status == "found"
    assert result.applications is not None
    assert len(result.applications) == 1
```

(`_paper`, `_claim`, `NEAR` mirror the helpers already defined in `tests/test_assessment_gap.py` — copy the same small helper functions into `test_assessment_applications.py` if that file doesn't already define its own equivalents; check the file's current content first before duplicating.)

Add to `tests/test_assessment_build.py`:

```python
def test_potential_applications_is_empty_list_not_null_when_relevant_papers_have_no_application(
    session_factory, embedder
) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "an unrelated method claim")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    # relevant paper WAS retrieved (near-exact title match) but stated no
    # application claim - this must be distinguishable from "nothing relevant
    # was retrieved at all", which stays None (see the pre-existing
    # test_assessment_defaults_unassessed_fields_rather_than_fabricating)
    assert assessment.potential_applications == []
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_applications.py tests/test_assessment_build.py -v`
Expected: FAIL — `ApplicationsResult` has no `status` field yet; `build.py` still collapses both cases to `None`.

- [x] **Step 3: Implement `applications.py`**

```python
@dataclass
class ApplicationsResult:
    applications: list[ApplicationRecord] | None
    evidence_ids: list[uuid.UUID]
    status: str = "not_assessed"  # "not_assessed" | "no_evidence" | "found"


_NOT_ASSESSED_RESULT = ApplicationsResult(applications=None, evidence_ids=[], status="not_assessed")
_NO_EVIDENCE_RESULT = ApplicationsResult(applications=[], evidence_ids=[], status="no_evidence")


def assess_applications(session: Session, papers_by_distance: list[PaperWithDistance]) -> ApplicationsResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns)."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    if not relevant_paper_ids:
        return _NOT_ASSESSED_RESULT

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "applications",
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return _NO_EVIDENCE_RESULT

    by_paper_id: dict[uuid.UUID, list[tuple[str, uuid.UUID, str]]] = {}
    for paper_id, text, evidence_id, title in rows:
        by_paper_id.setdefault(paper_id, []).append((text, evidence_id, title))

    applications: list[ApplicationRecord] = []
    evidence_ids: list[uuid.UUID] = []
    for paper_id in relevant_paper_ids:
        for text, evidence_id, title in by_paper_id.get(paper_id, []):
            applications.append(
                ApplicationRecord(application=text, source_paper=title, paper_id=paper_id, evidence_id=evidence_id)
            )
            evidence_ids.append(evidence_id)

    return ApplicationsResult(applications=applications, evidence_ids=evidence_ids, status="found")
```

Remove the old module-level `_EMPTY_RESULT` (replaced by the two new sentinels above).

- [x] **Step 4: Wire the distinction into `build.py`**

In `src/researchbridge/assessment/build.py`, replace lines 90-101:

```python
        potential_applications=(
            [
                {
                    "application": app.application,
                    "source_paper": app.source_paper,
                    "paper_id": str(app.paper_id),
                }
                for app in applications.applications
            ]
            if applications.applications
            else None
        ),
```

with:

```python
        potential_applications=(
            [
                {
                    "application": app.application,
                    "source_paper": app.source_paper,
                    "paper_id": str(app.paper_id),
                }
                for app in applications.applications
            ]
            if applications.status == "found"
            else ([] if applications.status == "no_evidence" else None)
        ),
```

- [x] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_assessment_applications.py tests/test_assessment_build.py -v`
Expected: PASS — including the pre-existing `test_assessment_defaults_unassessed_fields_rather_than_fabricating`, which asserts on other fields and doesn't check `potential_applications` directly (confirm via the run; if it does assert `potential_applications is None`, that's still correct since that test's scenario has zero relevant papers retrieved, which is the `not_assessed`/`None` case).

- [x] **Step 6: Commit**

```bash
git add src/researchbridge/assessment/applications.py src/researchbridge/assessment/build.py tests/test_assessment_applications.py tests/test_assessment_build.py
git commit -m "feat(assessment): distinguish 'no relevant papers' from 'no stated application'"
```

---

## Task 7: Dimension-aware risk coverage gaps (without fabricating risks)

**Files:**
- Modify: `src/researchbridge/assessment/risks.py`
- Test: `tests/test_assessment_risks.py`

**Interfaces:**
- Consumes: `DimensionCoverage` (Task 2).
- Produces: `RisksResult` gains `coverage_gaps: list[str]` (dimension labels whose status is `not_found` or `weak_evidence` among the *same relevant paper set* risks.py already gates on). `assess_risks()` gains optional `dimension_coverages: list[DimensionCoverage] | None = None`. The existing `text` field (the actual extracted risk statements) is completely unchanged in how it's built — `coverage_gaps` is additive and structurally separate, never merged into `text`.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_assessment_risks.py`:

```python
from researchbridge.assessment.coverage import DimensionCoverage


def test_coverage_gaps_default_to_empty_when_not_provided(session_factory) -> None:
    session = session_factory()
    result = assess_risks(session, [])
    session.close()
    assert result.coverage_gaps == []


def test_coverage_gaps_lists_not_found_and_weak_evidence_dimensions(session_factory) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "does not handle streaming data")
    session.commit()

    coverages = [
        DimensionCoverage(dimension="concept drift", status="not_found"),
        DimensionCoverage(dimension="class imbalance", status="weak_evidence", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="federated learning", status="established", evidence_ids=[uuid.uuid4()]),
    ]

    result = assess_risks(session, [(paper.id, NEAR)], dimension_coverages=coverages)

    session.close()
    assert result.coverage_gaps == ["concept drift", "class imbalance"]


def test_coverage_gaps_never_appear_inside_the_risk_text_itself(session_factory) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "limitations", "does not handle streaming data")
    session.commit()

    coverages = [DimensionCoverage(dimension="concept drift", status="not_found")]
    result = assess_risks(session, [(paper.id, NEAR)], dimension_coverages=coverages)

    session.close()
    # the coverage gap must never be silently promoted into a risk statement
    assert "concept drift" not in (result.text or "")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_risks.py -v`
Expected: FAIL — `assess_risks()` doesn't accept `dimension_coverages`, `RisksResult` has no `coverage_gaps`.

- [x] **Step 3: Implement the change**

In `src/researchbridge/assessment/risks.py`:

```python
from researchbridge.assessment.coverage import DimensionCoverage

...

_GAP_STATUSES = frozenset({"not_found", "weak_evidence"})


@dataclass
class RisksResult:
    text: str | None
    evidence_ids: list[uuid.UUID]
    coverage_gaps: list[str] = field(default_factory=list)


def assess_risks(
    session: Session,
    papers_by_distance: list[PaperWithDistance],
    dimension_coverages: list[DimensionCoverage] | None = None,
) -> RisksResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns).

    dimension_coverages (optional): from assessment/coverage.py. Used ONLY
    to populate coverage_gaps - a structurally separate list of dimension
    labels with little/no supporting literature. This is an evidence-
    coverage OBSERVATION, never a risk statement: a missing dimension is
    never auto-promoted into text, which stays 100% sourced from papers'
    own explicit limitations claims, exactly as before this parameter
    existed."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    coverage_gaps = [c.dimension for c in (dimension_coverages or []) if c.status in _GAP_STATUSES]

    if not relevant_paper_ids:
        return RisksResult(text=None, evidence_ids=[], coverage_gaps=coverage_gaps)

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "limitations",
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return RisksResult(text=None, evidence_ids=[], coverage_gaps=coverage_gaps)

    by_paper_id: dict[uuid.UUID, list[tuple[str, uuid.UUID, str]]] = {}
    for paper_id, text, evidence_id, title in rows:
        by_paper_id.setdefault(paper_id, []).append((text, evidence_id, title))

    lines: list[str] = []
    evidence_ids: list[uuid.UUID] = []
    for paper_id in relevant_paper_ids:
        for text, evidence_id, title in by_paper_id.get(paper_id, []):
            lines.append(f'- {title}: "{text}"')
            evidence_ids.append(evidence_id)

    return RisksResult(text="\n".join(lines), evidence_ids=evidence_ids, coverage_gaps=coverage_gaps)
```

Add `field` to the existing `from dataclasses import dataclass` import line.

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_risks.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/risks.py tests/test_assessment_risks.py
git commit -m "feat(assessment): surface dimension coverage gaps alongside risks, never as fabricated risks"
```

---

## Task 8: Inspectable recommendation reasoning

**Files:**
- Modify: `src/researchbridge/assessment/recommendation.py`
- Test: `tests/test_assessment_recommendation.py`

**Interfaces:** `RecommendationResult` gains `reasoning: str`. The decision table producing `recommendation`/`confidence` is **completely unchanged** — this task only adds an explanatory string built from the same inputs the function already receives, plus one new optional parameter.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_assessment_recommendation.py`:

```python
def test_reasoning_names_all_three_signals() -> None:
    result = assess_recommendation(
        novelty_level="medium", research_gap_text="a real gap", technical_feasibility_level="high"
    )
    assert "novelty" in result.reasoning.lower()
    assert "gap" in result.reasoning.lower()
    assert "feasibility" in result.reasoning.lower()


def test_reasoning_says_when_a_signal_was_not_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed", research_gap_text=None, technical_feasibility_level="medium"
    )
    lowered = result.reasoning.lower()
    assert "novelty" in lowered
    assert "not assessed" in lowered or "not_assessed" in lowered


def test_reasoning_mentions_recommendation_and_confidence() -> None:
    result = assess_recommendation(
        novelty_level="low", research_gap_text="a real gap", technical_feasibility_level="high"
    )
    assert result.recommendation.lower() in result.reasoning.lower()
    assert result.confidence.lower() in result.reasoning.lower()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_recommendation.py -v`
Expected: FAIL — `RecommendationResult` has no `reasoning` field.

- [x] **Step 3: Implement the change**

In `src/researchbridge/assessment/recommendation.py`:

```python
@dataclass
class RecommendationResult:
    recommendation: str
    confidence: str  # high | medium | low
    reasoning: str


def assess_recommendation(
    novelty_level: str,
    research_gap_text: str | None,
    technical_feasibility_level: str,
) -> RecommendationResult:
    novelty_assessed = novelty_level != "not_assessed"
    gap_found = research_gap_text is not None
    feasibility_assessed = technical_feasibility_level != "not_assessed"
    assessed_count = sum([novelty_assessed, gap_found, feasibility_assessed])

    if assessed_count == 0:
        recommendation = "INSUFFICIENT EVIDENCE"
    elif (
        novelty_level in _ASSESSED_NOVELTY_LEVELS
        and gap_found
        and technical_feasibility_level in ("medium", "high")
    ):
        recommendation = "HIGH PRIORITY"
    elif novelty_level == "low":
        recommendation = "LOW PRIORITY"
    elif assessed_count == 1:
        recommendation = "REQUIRES HUMAN REVIEW"
    else:
        recommendation = "MEDIUM PRIORITY"

    if assessed_count == 3:
        confidence = "high"
    elif assessed_count == 2:
        confidence = "medium"
    else:
        confidence = "low"

    reasoning = _build_reasoning(
        novelty_level=novelty_level,
        gap_found=gap_found,
        technical_feasibility_level=technical_feasibility_level,
        recommendation=recommendation,
        confidence=confidence,
    )

    return RecommendationResult(recommendation=recommendation, confidence=confidence, reasoning=reasoning)


def _build_reasoning(
    *, novelty_level: str, gap_found: bool, technical_feasibility_level: str, recommendation: str, confidence: str
) -> str:
    novelty_line = (
        f"Novelty signal: {novelty_level}"
        if novelty_level != "not_assessed"
        else "Novelty signal: not assessed (insufficient retrieved evidence)"
    )
    gap_line = (
        "Research gap evidence: a gap was found in the retrieved literature"
        if gap_found
        else "Research gap evidence: none found or not assessed"
    )
    feasibility_line = (
        f"Technical grounding: {technical_feasibility_level}"
        if technical_feasibility_level != "not_assessed"
        else "Technical grounding: not assessed (insufficient retrieved evidence)"
    )
    return (
        f"{novelty_line}\n{gap_line}\n{feasibility_line}\n\n"
        f"Recommendation: {recommendation}\nConfidence: {confidence}"
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_recommendation.py -v`
Expected: PASS — including every pre-existing test in the file (all assert only on `recommendation`/`confidence`, unaffected by the new field).

- [x] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/recommendation.py tests/test_assessment_recommendation.py
git commit -m "feat(assessment): make the recommendation chain inspectable via a reasoning field"
```

---

## Task 9: Wire the coverage layer into `build_assessment`

**Files:**
- Modify: `src/researchbridge/assessment/build.py`
- Test: `tests/test_assessment_build.py`

**Interfaces:**
- Consumes: `extract_dimensions` (Task 1), `compute_dimension_coverage` (Task 2), the updated `assess_novelty`/`assess_risks` signatures (Tasks 4, 7), the updated `GapAssessmentResult.status`/`ApplicationsResult.status` (Tasks 5, 6), the updated `RecommendationResult.reasoning` (Task 8).
- `build_assessment()`'s own signature is unchanged (`session, research_input_id, embedder, top_k=10`).

- [x] **Step 1: Write the failing integration tests**

Add to `tests/test_assessment_build.py`:

```python
def test_novelty_reasoning_includes_dimension_coverage_when_dimensions_are_extracted(
    session_factory, embedder
) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "federated learning for fraud detection across banks")
    _claim(session, paper, "method", "a federated learning method for fraud detection")
    ri = _research_input(
        session,
        "A federated learning system for detecting financial fraud across multiple banks "
        "while remaining robust to concept drift and class imbalance.",
    )
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert "Dimension coverage:" in assessment.novelty_reasoning


def test_recommendation_reasoning_is_persisted_and_derivable_at_read_time(session_factory, embedder) -> None:
    # recommendation reasoning is NOT a new column - it's derived at read
    # time from novelty_level/novelty_reasoning/research_gap_text/
    # technical_feasibility_level/recommendation/confidence, all of which
    # ARE already persisted. This test confirms build_assessment still
    # populates every one of those source fields (the narrative builder
    # itself is exercised in Task 10's export/API test).
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level is not None
    assert assessment.technical_feasibility_level is not None
    assert assessment.recommendation is not None
    assert assessment.confidence is not None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_build.py -v`
Expected: The first new test FAILs (`build_assessment` doesn't compute dimensions/coverage yet, so `novelty_reasoning` has no "Dimension coverage:" block). The second should already pass (it only checks pre-existing behavior) — confirms it's a true regression guard, not a new requirement.

- [x] **Step 3: Wire dimensions and coverage into `build_assessment`**

In `src/researchbridge/assessment/build.py`, add imports:

```python
from researchbridge.assessment.coverage import compute_dimension_coverage
from researchbridge.assessment.dimensions import extract_dimensions
```

Replace the body from `results = search_by_text(...)` through the `recommendation = assess_recommendation(...)` call:

```python
    results = search_by_text(session, query_text, embedder, top_k)
    papers_with_claims = [(paper, distance, _claims_for_paper(session, paper.id)) for paper, distance in results]

    dimensions = extract_dimensions(query_text)
    dimension_coverages = compute_dimension_coverage(
        dimensions,
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims],
        embedder,
    )

    existing_solutions = build_existing_solutions(
        [(paper.title, claims) for paper, _distance, claims in papers_with_claims if claims]
    )

    novelty = assess_novelty(
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims],
        dimension_coverages=dimension_coverages,
    )

    retrieved_paper_uuids = [paper.id for paper, _distance, _claims in papers_with_claims]
    papers_by_distance = [(paper.id, distance) for paper, distance, _claims in papers_with_claims]
    gap = assess_research_gap(session, papers_by_distance, embedder)
    applications = assess_applications(session, papers_by_distance)
    feasibility = assess_technical_feasibility(session, papers_by_distance)
    opportunities = assess_opportunities(session, papers_by_distance)
    risks = assess_risks(session, papers_by_distance, dimension_coverages=dimension_coverages)
    external_validation_needed = assess_external_validation(has_applications=bool(applications.applications))
    recommendation = assess_recommendation(
        novelty_level=novelty.level,
        # only count the gap as "found" for recommendation purposes if it's
        # closely grounded - the report field still shows gap.text regardless
        research_gap_text=(gap.text if gap.is_closely_grounded else None),
        technical_feasibility_level=feasibility.level,
    )
```

Update the `potential_applications=` assembly per Task 6 Step 4 if not already applied (it's the same edit; if Task 6 was completed first, this is already in place — just confirm it's still there after this diff).

Update `research_gap_source` assembly to carry the new not_assessed/not_found vocabulary. Change:

```python
        research_gap_source=gap.source,
```

to:

```python
        research_gap_source=(
            gap.source
            if gap.status == "found"
            else ("no_relevant_evidence" if gap.status == "not_assessed" else "checked_no_gap_found")
        ),
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_build.py -v`
Expected: PASS — including every pre-existing test in the file. In particular, re-check `test_assessment_defaults_unassessed_fields_rather_than_fabricating` (line 143 in the original file): it asserts `assessment.research_gap_text is None`, which is still true (only `research_gap_source`'s value changed, from `None` to `"no_relevant_evidence"` in this no-relevant-papers scenario) — if the test doesn't check `research_gap_source`, it passes unmodified; if a reviewer wants that source value asserted too, add `assert assessment.research_gap_source == "no_relevant_evidence"` to that test as a tightening, not a requirement of this step.

- [x] **Step 5: Run the full test suite to check for regressions**

Run: `pytest tests/ -v -k "assessment"`
Expected: PASS across every assessment-related test file.

- [x] **Step 6: Commit**

```bash
git add src/researchbridge/assessment/build.py tests/test_assessment_build.py
git commit -m "feat(assessment): wire dimension coverage into build_assessment orchestration"
```

---

## Task 10: Surface the new signals in the API, export, and frontend report

**Files:**
- Modify: `src/researchbridge/api/schemas.py`
- Create: `src/researchbridge/assessment/narrative.py`
- Modify: `src/researchbridge/assessment/export.py`
- Modify: `frontend/components/AssessmentReport.tsx`
- Modify: `frontend/lib/assessmentApi.ts`
- Test: `tests/test_assessment_narrative.py`
- Test: `tests/test_assessment_export.py`
- Test: `tests/test_assessment_api.py`

**Interfaces:**
- Produces: `build_recommendation_narrative(assessment: ResearchAssessmentOut) -> str` — a pure, read-time function computed entirely from already-persisted fields (`novelty_level`, `novelty_reasoning`, `research_gap_text`, `research_gap_source`, `technical_feasibility_level`, `technical_feasibility_reasoning`, `recommendation`, `confidence`). No new persisted field.

- [x] **Step 1: Write the failing narrative test**

Create `tests/test_assessment_narrative.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from researchbridge.api.schemas import ResearchAssessmentOut, ResearchInputOut
from researchbridge.assessment.narrative import build_recommendation_narrative


def _assessment(**overrides) -> ResearchAssessmentOut:
    defaults = dict(
        id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc),
        research_input=ResearchInputOut(
            id=uuid.uuid4(), input_type="idea", raw_text="an idea", created_at=datetime.now(timezone.utc)
        ),
        status="completed",
        retrieved_paper_ids=[],
        comparison_summary=None,
        novelty_level="medium",
        novelty_reasoning="The closest retrieved paper is moderately related.\n\nDimension coverage:\n  concept drift -> not_found",
        research_gap_text="a stated gap",
        research_gap_source="input_specific",
        candidate_gap_id=None,
        potential_applications=None,
        technical_feasibility_level="high",
        technical_feasibility_reasoning="Two relevant papers document applicable methods.",
        potential_opportunities=None,
        risks_and_limitations=None,
        external_validation_needed="Always required.",
        recommendation="HIGH PRIORITY",
        confidence="high",
        human_reviewed=False,
        evidence=[],
    )
    defaults.update(overrides)
    return ResearchAssessmentOut(**defaults)


def test_narrative_includes_all_four_signal_lines() -> None:
    narrative = build_recommendation_narrative(_assessment())

    assert "Novelty signal:" in narrative
    assert "Research gap evidence:" in narrative
    assert "Technical grounding:" in narrative
    assert "Recommendation:" in narrative


def test_narrative_carries_the_dimension_coverage_block_through_from_novelty_reasoning() -> None:
    narrative = build_recommendation_narrative(_assessment())
    assert "concept drift -> not_found" in narrative


def test_narrative_distinguishes_not_assessed_from_checked_no_gap_found() -> None:
    not_assessed = build_recommendation_narrative(
        _assessment(research_gap_text=None, research_gap_source="no_relevant_evidence")
    )
    not_found = build_recommendation_narrative(
        _assessment(research_gap_text=None, research_gap_source="checked_no_gap_found")
    )

    assert not_assessed != not_found
    assert "insufficient" in not_assessed.lower() or "not enough" in not_assessed.lower()
    assert "no gap" in not_found.lower() or "none found" in not_found.lower()


def test_narrative_reports_final_recommendation_and_confidence() -> None:
    narrative = build_recommendation_narrative(_assessment(recommendation="LOW PRIORITY", confidence="low"))
    assert "LOW PRIORITY" in narrative
    assert "low" in narrative.lower()
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_assessment_narrative.py -v`
Expected: FAIL — `researchbridge.assessment.narrative` doesn't exist yet.

- [x] **Step 3: Implement `narrative.py`**

```python
"""Read-time recommendation narrative (blueprint constraint: the
recommendation chain must be inspectable, without adding a new persisted
column - see docs/superpowers/plans/2026-08-29-dimension-aware-assessment-
coverage.md, "Migration implications").

Purely derived from fields ResearchAssessment already persists. The
per-dimension coverage table lives inside novelty_reasoning's own text
(assessment/novelty.py appends it there); this function doesn't recompute
it, just stitches the already-persisted pieces into one readable block.
"""

from __future__ import annotations

from researchbridge.api.schemas import ResearchAssessmentOut

_GAP_SOURCE_LABELS = {
    "reused_candidate_gap": "a reviewed candidate gap was reused",
    "input_specific": "found for this input",
    "no_relevant_evidence": "not assessed - insufficient relevant evidence was retrieved to investigate",
    "checked_no_gap_found": "none found - relevant literature was checked, no gap surfaced",
}


def build_recommendation_narrative(assessment: ResearchAssessmentOut) -> str:
    novelty_line = f"Novelty signal: {assessment.novelty_level}"
    if assessment.novelty_reasoning:
        novelty_line += f"\n{assessment.novelty_reasoning}"

    if assessment.research_gap_text:
        gap_line = f"Research gap evidence: {assessment.research_gap_text}"
    else:
        gap_line = f"Research gap evidence: {_GAP_SOURCE_LABELS.get(assessment.research_gap_source, 'not assessed')}"

    feasibility_line = f"Technical grounding: {assessment.technical_feasibility_level}"
    if assessment.technical_feasibility_reasoning:
        feasibility_line += f"\n{assessment.technical_feasibility_reasoning}"

    recommendation_line = f"Recommendation: {assessment.recommendation}\nConfidence: {assessment.confidence}"

    return "\n\n".join([novelty_line, gap_line, feasibility_line, recommendation_line])
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_assessment_narrative.py -v`
Expected: PASS

- [x] **Step 5: Write the failing export tests**

Add to `tests/test_assessment_export.py` (near the existing "No gap was found" assertions at lines ~107-130):

```python
def test_export_distinguishes_not_assessed_gap_from_checked_no_gap_found() -> None:
    not_assessed = _build_assessment_out(research_gap_text=None, research_gap_source="no_relevant_evidence")
    not_found = _build_assessment_out(research_gap_text=None, research_gap_source="checked_no_gap_found")

    sections_not_assessed = build_report_sections(not_assessed)
    sections_not_found = build_report_sections(not_found)

    gap_section_a = next(s for s in sections_not_assessed if s.label == "Research gap")
    gap_section_b = next(s for s in sections_not_found if s.label == "Research gap")

    assert gap_section_a.unassessed_reason != gap_section_b.unassessed_reason
    assert "insufficient" in gap_section_a.unassessed_reason.lower()
    assert "no gap" in gap_section_b.unassessed_reason.lower() or "none" in gap_section_b.unassessed_reason.lower()


def test_export_distinguishes_applications_not_assessed_from_no_evidence() -> None:
    not_assessed = _build_assessment_out(potential_applications=None)
    no_evidence = _build_assessment_out(potential_applications=[])

    sections_not_assessed = build_report_sections(not_assessed)
    sections_no_evidence = build_report_sections(no_evidence)

    app_section_a = next(s for s in sections_not_assessed if s.label == "Potential applications")
    app_section_b = next(s for s in sections_no_evidence if s.label == "Potential applications")

    assert app_section_a.unassessed_reason != app_section_b.unassessed_reason
```

(`_build_assessment_out` should already exist as a helper in `test_assessment_export.py` given the existing tests at lines 33/66 construct `ResearchAssessmentOut`-shaped fixtures directly — if no such shared helper exists yet, check the file's current fixture pattern first and factor one out rather than duplicating the full field list four times.)

- [x] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_assessment_export.py -v`
Expected: FAIL — `build_report_sections` doesn't yet vary `unassessed_reason` by `research_gap_source`/by empty-vs-null `potential_applications`.

- [x] **Step 7: Implement the change in `export.py`**

In `src/researchbridge/assessment/export.py`, update `build_report_sections`:

```python
    research_gap_unassessed_reason = _GAP_UNASSESSED_REASONS.get(
        assessment.research_gap_source, _GAP_UNASSESSED_REASONS[None]
    )

    applications_body = None
    applications_unassessed_reason = _APPLICATIONS_UNASSESSED_REASONS["not_assessed"]
    if assessment.potential_applications:
        applications_body = "\n".join(
            f"- {app['application']} (source: {app['source_paper']})" for app in assessment.potential_applications
        )
    elif assessment.potential_applications == []:
        applications_unassessed_reason = _APPLICATIONS_UNASSESSED_REASONS["no_evidence"]
```

and change the `ReportSection` entries for research gap and applications to use these instead of the hardcoded strings:

```python
        ReportSection(
            label="Research gap",
            body=research_gap_body,
            unassessed_reason=research_gap_unassessed_reason,
            evidence=by_role.get("research_gap", []),
        ),
        ...
        ReportSection(
            label="Potential applications",
            body=applications_body,
            unassessed_reason=applications_unassessed_reason,
            evidence=by_role.get("application", []),
        ),
```

Add the lookup tables near the top of the file, alongside the existing `OPPORTUNITIES_REASON` constant:

```python
_GAP_UNASSESSED_REASONS = {
    "no_relevant_evidence": (
        "Not assessed - insufficient relevant evidence was retrieved for this input to "
        "investigate whether a research gap exists."
    ),
    "checked_no_gap_found": "No gap was found in the retrieved literature for this input.",
    None: "No gap was found in the retrieved literature for this input.",
}

_APPLICATIONS_UNASSESSED_REASONS = {
    "not_assessed": "No relevant paper was retrieved for this input, so applications could not be assessed.",
    "no_evidence": "No retrieved paper stated an application.",
}
```

- [x] **Step 8: Run export tests to verify they pass**

Run: `pytest tests/test_assessment_export.py -v`
Expected: PASS. Confirm the pre-existing tests at lines ~107/129 (`"No gap was found" in text`, `"No retrieved paper stated an application" in text`) still pass — they exercise the `research_gap_source=None`/`potential_applications=None` defaults, which map to the same fallback strings as before.

- [x] **Step 9: Update the frontend report**

In `frontend/lib/assessmentApi.ts`, find the `ResearchAssessment` type's `research_gap_source` field and widen its documented (TypeScript string literal, if typed as a union) value set to include the two new sentinels — check the current type definition first:

```bash
grep -n "research_gap_source" frontend/lib/assessmentApi.ts
```

If it's typed as `string | null`, no type change is needed, only the rendering logic below. If it's a string-literal union, add `"no_relevant_evidence" | "checked_no_gap_found"` to it.

In `frontend/components/AssessmentReport.tsx`, replace the "research gap" field block (currently lines 135-150):

```tsx
      <Field label="research gap" evidence={byRole.get("research_gap")}>
        {assessment.research_gap_text ? (
          <>
            <Prose text={assessment.research_gap_text} />
            {assessment.research_gap_source && (
              <p className="eyebrow mt-3">
                {assessment.research_gap_source === "reused_candidate_gap"
                  ? "reused a reviewed candidate gap"
                  : "found for this input"}
              </p>
            )}
          </>
        ) : (
          <Unassessed
            reason={
              assessment.research_gap_source === "no_relevant_evidence"
                ? "Not assessed - insufficient relevant evidence was retrieved for this input to investigate whether a research gap exists."
                : "No gap was found in the retrieved literature for this input."
            }
          />
        )}
      </Field>
```

and replace the "potential applications" field block (currently lines 152-169):

```tsx
      <Field label="potential applications" evidence={byRole.get("application")}>
        {assessment.potential_applications?.length ? (
          <ul className="space-y-3">
            {assessment.potential_applications.map((app, i) => (
              <li key={i}>
                <p className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed">
                  {app.application}
                </p>
                <Link href={`/papers/${app.paper_id}`} className="eyebrow mt-1 inline-block hover:text-[var(--ink)]">
                  {app.source_paper}
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <Unassessed
            reason={
              assessment.potential_applications === null
                ? "No relevant paper was retrieved for this input, so applications could not be assessed."
                : "No retrieved paper stated an application."
            }
          />
        )}
      </Field>
```

Add a new field block after "external validation needed" (end of the article, before the closing `</article>`), rendering the recommendation narrative — the simplest zero-new-endpoint approach is to compute it client-side with the same four already-fetched fields, mirroring `narrative.py`'s logic:

```tsx
      <Field label="recommendation reasoning" gradeable={false}>
        <Preformatted
          text={[
            `Novelty signal: ${assessment.novelty_level}`,
            assessment.novelty_reasoning ?? "",
            "",
            assessment.research_gap_text
              ? `Research gap evidence: ${assessment.research_gap_text}`
              : `Research gap evidence: ${
                  assessment.research_gap_source === "no_relevant_evidence"
                    ? "not assessed - insufficient relevant evidence"
                    : "none found - relevant literature was checked"
                }`,
            "",
            `Technical grounding: ${assessment.technical_feasibility_level}`,
            assessment.technical_feasibility_reasoning ?? "",
            "",
            `Recommendation: ${assessment.recommendation ?? "not assessed"}`,
            `Confidence: ${assessment.confidence ?? "not assessed"}`,
          ]
            .filter((line) => line !== undefined)
            .join("\n")}
        />
      </Field>
```

This duplicates `narrative.py`'s stitching logic in TypeScript rather than adding a new API round-trip for a string trivially computable from fields the page already has — acceptable for this first iteration; if the duplication drifts, a follow-up can expose `GET /api/assessments/{id}` returning the narrative pre-built server-side instead.

- [x] **Step 10: Manually verify the frontend renders correctly**

Start the dev server and view a real assessment report:

```bash
npm --prefix frontend run dev
```

Open an existing assessment in the browser, confirm: the "research gap" and "potential applications" sections show the new distinct messages when applicable, and the new "recommendation reasoning" section renders the four-line block without layout breakage. Screenshot before considering this step done.

- [x] **Step 11: Commit**

```bash
git add src/researchbridge/assessment/narrative.py src/researchbridge/assessment/export.py frontend/components/AssessmentReport.tsx frontend/lib/assessmentApi.ts tests/test_assessment_narrative.py tests/test_assessment_export.py
git commit -m "feat(assessment): render inspectable recommendation narrative and distinct not_assessed/not_found messaging"
```

---

## Task 11: End-to-end acceptance test (fraud/federated-learning idea)

**Files:**
- Create: `tests/test_assessment_dimension_coverage_e2e.py`

**Interfaces:** Exercises `build_assessment` only — no new production code in this task, pure verification.

This test validates **structure and grounding guarantees**, not a predetermined scientific conclusion — per the constraint, it must not assert a specific `recommendation` value or a specific novelty level, since that depends on what's actually in the corpus.

- [x] **Step 1: Write the acceptance test**

```python
from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.build import build_assessment
from researchbridge.db.models import (
    EMBEDDING_DIM,
    Embedding,
    Evidence,
    ExtractedClaim,
    Paper,
    ResearchAssessmentEvidence,
    ResearchInput,
)
from researchbridge.embedding.pipeline import EMBEDDING_TYPE

FRAUD_IDEA = (
    "Can we build a privacy-preserving federated learning system for detecting "
    "financial fraud across multiple banks without sharing raw transaction data, "
    "while remaining robust to concept drift and highly imbalanced fraud classes?"
)


def _hash_to_unit_vector(text: str) -> list[float]:
    import hashlib

    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


class FakeEmbedder:
    model_name = "fake-e2e-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_unit_vector(t) for t in texts]


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector))
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str) -> uuid.UUID:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )
    return evidence.id


def test_fraud_idea_end_to_end_structural_guarantees(session_factory, embedder) -> None:
    session = session_factory()

    p1 = _paper(session, embedder, "p1", FRAUD_IDEA)  # near-identical, drives retrieval
    _claim(session, p1, "method", "We propose a federated learning approach for cross-bank fraud detection.")
    _claim(session, p1, "results", "Our federated approach achieves 94% detection accuracy.")
    _claim(session, p1, "main_contribution", "We show that federated aggregation preserves privacy across banks.")
    _claim(session, p1, "limitations", "The method assumes a stationary fraud distribution and does not handle concept drift.")

    p2 = _paper(session, embedder, "p2", "privacy preserving fraud detection across banks")
    _claim(session, p2, "problem", "Sharing raw transaction data across banks raises privacy concerns.")
    _claim(session, p2, "limitations", "Highly imbalanced fraud classes remain a challenge for this approach.")

    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=FRAUD_IDEA)
    session.add(ri)
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=10)

    # retrieves relevant papers
    assert str(p1.id) in assessment.retrieved_paper_ids
    assert str(p2.id) in assessment.retrieved_paper_ids

    # identifies multiple dimensions of the idea + produces per-dimension coverage
    assert "Dimension coverage:" in assessment.novelty_reasoning
    coverage_block = assessment.novelty_reasoning.split("Dimension coverage:")[1]
    coverage_lines = [line for line in coverage_block.splitlines() if "->" in line]
    assert len(coverage_lines) >= 2

    # does not classify results or main_contribution as problems
    assert assessment.comparison_summary is not None
    assert "94% detection accuracy" not in _section(assessment.comparison_summary, "Problems already addressed")
    assert "federated aggregation preserves privacy" not in _section(
        assessment.comparison_summary, "Problems already addressed"
    )

    # does not treat generic business/application language as a research gap:
    # gap.py only ever sources research_gap_text from an explicit research_gap-
    # type claim, a reused approved candidate gap, or a cross-paper limitation/
    # research_gap cluster - never from problem/method/results/applications
    # claims, which is what "generic business language" would show up as.
    # No research_gap-type claim exists in this fixture, so the gap must be
    # either None (not_found/not_assessed) or explicitly source-labeled.
    assert assessment.research_gap_source in (None, "no_relevant_evidence", "checked_no_gap_found", "input_specific")
    if assessment.research_gap_text is not None:
        assert assessment.research_gap_source in ("input_specific", "reused_candidate_gap")

    # distinguishes explicit gaps from inferred/coverage-based observations:
    # research_gap_source, when a gap IS found, is never a bare "found" -
    # it names its own provenance
    if assessment.research_gap_text is not None:
        assert assessment.research_gap_source is not None

    # keeps evidence traceable to real source claims
    evidence_links = session.query(ResearchAssessmentEvidence).filter_by(research_assessment_id=assessment.id).all()
    real_evidence_ids = {
        row.id
        for row in session.query(Evidence).filter(Evidence.paper_id.in_([p1.id, p2.id])).all()
    }
    for link in evidence_links:
        assert link.evidence_id in real_evidence_ids

    # does not generate product opportunities
    assert assessment.potential_opportunities is None

    # produces an explainable recommendation - the reasoning is reconstructible
    # from already-persisted fields (see narrative.py, Task 10)
    assert assessment.recommendation is not None
    assert assessment.confidence is not None

    session.close()


def test_fraud_idea_honestly_returns_insufficient_evidence_with_an_unrelated_corpus(session_factory, embedder) -> None:
    session = session_factory()

    unrelated = _paper(session, embedder, "u1", "the history of medieval bread baking techniques")
    _claim(session, unrelated, "method", "We survey historical baking equipment across three centuries.")

    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=FRAUD_IDEA)
    session.add(ri)
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=10)

    session.close()
    # honest non-recommendation is an acceptable, expected outcome - this
    # test does NOT assert a specific novelty/recommendation verdict for
    # the fraud idea in general, only that an unrelated corpus can't force
    # a fabricated one
    assert assessment.recommendation in ("INSUFFICIENT EVIDENCE", "REQUIRES HUMAN REVIEW", "LOW PRIORITY", "MEDIUM PRIORITY")


def _section(comparison_summary: str, heading: str) -> str:
    if heading not in comparison_summary:
        return ""
    after = comparison_summary.split(heading, 1)[1]
    # a section ends at the next heading (blank-line-separated block) or EOF
    return after.split("\n\n", 1)[0]
```

- [x] **Step 2: Run the acceptance test**

Run: `pytest tests/test_assessment_dimension_coverage_e2e.py -v`
Expected: PASS, assuming Tasks 1-9 are complete. If it fails, the failure should point at exactly one of the structural guarantees above being violated — treat that as a real bug in the corresponding task's implementation, not a reason to weaken this test.

- [x] **Step 3: Commit**

```bash
git add tests/test_assessment_dimension_coverage_e2e.py
git commit -m "test(assessment): add end-to-end structural acceptance test for dimension coverage"
```

---

## Task 12: Live-corpus calibration pass (manual, not automated)

This task is deliberately not a pytest file — it's the same manual calibration step every threshold in this codebase (`NEAR_DISTANCE`/`FAR_DISTANCE` in `novelty.py`, the tightened threshold in `feasibility.py`, `DEFAULT_SIMILARITY_THRESHOLD` in `gaps/cluster.py`) was originally checked against before being trusted, per this codebase's own established practice.

- [x] **Step 1: Run `extract_dimensions` against 5-10 real research ideas**

Use a scratch script (not committed) to run `extract_dimensions()` from Task 1 against a handful of real ideas the corpus has (or ideas a user is likely to submit — the fraud/federated-learning example plus a few others spanning different domains in the corpus). Eyeball the output: are the phrases genuinely distinct concepts, or is the stopword list letting noise through (e.g. "we propose", "this paper")? Adjust `_STOPWORDS` in `dimensions.py` if so, and re-run Task 1's test suite to confirm no regression.

- [x] **Step 2: Run `compute_dimension_coverage` against the live corpus with the fraud idea**

Using a real `Embedder` (not the test fakes) against the actual database, run `build_assessment` for the fraud/federated-learning idea from Task 11 against whatever real papers exist in the corpus. Inspect the resulting `novelty_reasoning`'s dimension-coverage block: do the `established`/`partially_addressed`/`weak_evidence`/`not_found` statuses look right by manual judgment? In particular check `DIMENSION_MATCH_SIMILARITY = 0.30` in `coverage.py` — if real claim/dimension pairs that are obviously related score below it (false negatives -> everything reads `not_found`) or obviously unrelated pairs score above it (false positives -> everything reads `established`), adjust the constant.

- [x] **Step 3: Document the calibration**

Update `coverage.py`'s module docstring with a calibration note in the same style as `novelty.py`'s (lines 14-28) and `gaps/cluster.py`'s (lines 15-26): what was checked, what the raw similarity distribution looked like, why the final threshold was picked. This is a documentation-only change to a comment block, not new logic — re-run `pytest tests/test_assessment_coverage.py -v` afterward only to confirm nothing was accidentally broken while editing the file.

- [x] **Step 4: Sanity-check the novelty aggregate-rule thresholds (0.7/0.5/0.3) from Task 4**

Using the same live-corpus session, try 3-5 different ideas spanning "clearly already well-covered by the corpus" to "clearly novel relative to the corpus" and confirm the aggregate rule's `low`/`medium`/`high` boundaries feel right. Adjust the constants in `novelty.py`'s `_aggregate_from_coverage` if not, and add the same kind of calibration note to that function's docstring.

- [x] **Step 5: Commit the documentation updates**

```bash
git add src/researchbridge/assessment/coverage.py src/researchbridge/assessment/novelty.py src/researchbridge/assessment/dimensions.py
git commit -m "docs(assessment): document live-corpus calibration for dimension coverage thresholds"
```

---

## Self-review notes (for the plan author, not a task to execute)

- **Spec coverage:** every numbered point from the approval message maps to a task — dimension definition (pre-Task-1 section + Task 1), in-memory-only representation (Task 2 + "Migration implications"), evidence-grounding invariant (Task 2's construction + Task 11's traceability assertion), novelty aggregate rule (Task 4), inspectable recommendation chain (Tasks 8, 10), not_found vs not_assessed (Tasks 5, 6, 10), applications stay extractive (Task 6, untouched core query logic), risks don't fabricate (Task 7), existing_solutions bug fix (Task 3), end-to-end acceptance test (Task 11), minimal footprint / no migration (throughout + explicit table).
- **Placeholder scan:** no task contains "TBD"/"handle appropriately"/etc. — every step has literal code or a literal shell command.
- **Type consistency:** `DimensionCoverage`, `IdeaDimension`, `EvidencedPaper`, `ClaimRecord` are defined once (Tasks 1-2) and reused with identical field names throughout Tasks 4, 7, 9, 11 — cross-checked against each task's "Consumes" line.
