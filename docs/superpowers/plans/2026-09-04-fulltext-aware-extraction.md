# Full-Text-Aware Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `HeuristicExtractor`, `SemanticExtractor`, and `HybridExtractor` to draw claims from `paper_fulltext.sections` when available, instead of only ever reading `paper.abstract`.

**Architecture:** `Extractor.extract()` gains a second parameter, `sections: dict[str, str]`. A new shared helper (`extraction/sections.py`) resolves each Sec 25/28 field to its most likely full-text section(s) via a priority-ordered map (`FIELD_SECTIONS` in `extraction/base.py`), falling back to the abstract when a paper has no full text or no matching section. Each extractor keeps its existing abstract-only algorithm exactly as a fallback path, and layers full-text handling on top without changing behavior for the ~47,000 papers that don't have full text yet.

**Tech Stack:** Python, SQLAlchemy.

**Spec:** `docs/superpowers/specs/2026-09-04-fulltext-aware-extraction-design.md`

## Global Constraints

- No re-extraction of already-processed papers - this only affects new extraction runs.
- `StubExtractor` is untouched.
- A paper with `sections={}` (no full text) must produce byte-identical output to today's abstract-only extractors - every existing test's assertions stay valid once `sections={}` is added to the call.
- `HeuristicExtractor`'s full-text, named-section cue-phrase hits get `confidence="high"`; abstract-only hits stay `"medium"`/`"low"` exactly as today.
- `SemanticExtractor`'s confidence is NOT bumped for full-text hits - only the section is tagged; thresholds (`MIN_SIMILARITY`, `MEDIUM_CONFIDENCE_SIMILARITY`) are unchanged.
- `SemanticExtractor` must preserve its "one sentence proposes to one field" collision-avoidance: fields are grouped by their exact resolved sentence pool (same section, or the shared abstract pool) before the bipartite matching runs, never resolved field-by-field in isolation.
- Full text always wins over the abstract for a field when a full-text match exists for it.
- `FIELD_SECTIONS`' per-field section tuple is a priority order, not a pool - the first listed section present in `sections` is used; sections are never merged across the tuple for one field.

---

### Task 1: `extraction/base.py` - `FIELD_SECTIONS`, `ClaimCandidate.section`, `Extractor` signature

**Files:**
- Modify: `src/researchbridge/extraction/base.py`
- Test: `tests/test_extraction_base.py` (new file)

**Interfaces:**
- Produces: `FIELD_SECTIONS: dict[str, tuple[str, ...]]`, `ClaimCandidate` (now with `section: str = "abstract"`), `Extractor.extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]`. Every later task consumes these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extraction_base.py
from __future__ import annotations

from researchbridge.extraction.base import FIELD_SECTIONS, ClaimCandidate


def test_claim_candidate_defaults_section_to_abstract() -> None:
    candidate = ClaimCandidate(claim_type="method", claim_text="x", evidence_quote="x", confidence="medium")
    assert candidate.section == "abstract"


def test_claim_candidate_accepts_an_explicit_section() -> None:
    candidate = ClaimCandidate(
        claim_type="method", claim_text="x", evidence_quote="x", confidence="high", section="methods"
    )
    assert candidate.section == "methods"


def test_field_sections_covers_every_cue_phrase_field() -> None:
    # every field HeuristicExtractor's _CUE_PHRASES covers (all Sec 28
    # fields except "problem", which stays abstract-only by design) must
    # have a FIELD_SECTIONS entry, or it would silently never get a
    # full-text match at all
    expected_fields = {
        "research_question", "method", "dataset", "main_contribution",
        "results", "limitations", "applications", "research_gap",
    }
    assert expected_fields.issubset(FIELD_SECTIONS.keys())


def test_field_sections_values_are_nonempty_tuples_of_known_section_names() -> None:
    known_sections = {
        "abstract", "introduction", "related_work", "methods", "experiments",
        "results", "discussion", "limitations", "conclusion", "acknowledgments",
        "references", "body",
    }
    for field, section_tuple in FIELD_SECTIONS.items():
        assert len(section_tuple) > 0, f"{field} has an empty section tuple"
        assert set(section_tuple).issubset(known_sections), f"{field} references an unknown section"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_base.py -v`
Expected: FAIL with `ImportError: cannot import name 'FIELD_SECTIONS'`

- [ ] **Step 3: Implement**

Replace the full contents of `src/researchbridge/extraction/base.py`:

```python
"""Extractor interface: provider/model-agnostic claim extraction from a paper.

Any implementation - a local/free model, a mock/stub, or (optional, per the
blueprint's Free/Open-Source First constraint) a paid API - plugs in here
without the extraction pipeline needing to change. See ResearchBridge.md §29.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from researchbridge.db.models import Paper

# The Sec 25/28 field set, in reading order - reused wherever claims need a
# sensible display order (the API) rather than whatever order they happen
# to be stored/queried in.
CLAIM_TYPE_ORDER = (
    "problem",
    "research_question",
    "method",
    "dataset",
    "main_contribution",
    "results",
    "limitations",
    "research_gap",
    "applications",
)

# Sec 25/28 field -> the fulltext/parse.py section names most likely to
# actually state that field, in priority order (Sec 46's "full-text-aware
# extraction" design). A field's tuple is a priority list, not a pool:
# extraction/sections.py::sentences_for_field returns the FIRST listed
# section that's present in a paper's sections dict, never merging
# sentences across sections for one field. "problem" has no entry - its
# first-sentence-of-abstract fallback has no full-text equivalent worth
# inventing (see heuristic.py). "references"/"acknowledgments" are never a
# target - neither carries claim-bearing prose.
FIELD_SECTIONS: dict[str, tuple[str, ...]] = {
    "research_question": ("introduction",),
    "method": ("methods", "experiments"),
    "dataset": ("methods", "experiments"),
    "main_contribution": ("introduction", "conclusion"),
    "results": ("results",),
    "limitations": ("discussion", "limitations", "conclusion"),
    "research_gap": ("conclusion", "discussion", "limitations"),
    "applications": ("discussion", "conclusion", "introduction"),
}


@dataclass
class ClaimCandidate:
    claim_type: str
    claim_text: str
    evidence_quote: str
    confidence: str
    section: str = "abstract"
    """Which section evidence_quote was drawn from - "abstract" (the only
    value before Sec 46) or one of fulltext/parse.py's real section names.
    extraction/pipeline.py uses this to know what text to ground the quote
    against and what to persist as Evidence.section."""


class Extractor(Protocol):
    extraction_method: str
    model_version: str

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        """Return candidate claims for a paper.

        `sections` is this paper's paper_fulltext.sections (Sec 46), or {}
        when no full-text row exists yet - the common case until
        rb-fulltext-fetch processes more of the corpus. evidence_quote must
        be a verbatim substring of whatever section produced it
        (paper.abstract when candidate.section == "abstract", else
        sections[candidate.section]) - the pipeline verifies this before
        persisting anything and rejects candidates that fail.
        """
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/base.py tests/test_extraction_base.py
git commit -m "feat(extraction): add FIELD_SECTIONS and ClaimCandidate.section"
```

---

### Task 2: `extraction/sections.py` - `sentences_for_field`

**Files:**
- Create: `src/researchbridge/extraction/sections.py`
- Test: `tests/test_extraction_sections.py`

**Interfaces:**
- Consumes: `FIELD_SECTIONS` (Task 1), `split_sentences` (existing, `extraction/sentences.py`).
- Produces: `sentences_for_field(field: str, sections: dict[str, str]) -> list[tuple[str, str]]` - used by Tasks 3 and 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extraction_sections.py
from __future__ import annotations

from researchbridge.extraction.sections import sentences_for_field


def test_returns_sentences_from_the_first_present_target_section() -> None:
    sections = {"discussion": "First sentence here. Second one too.", "conclusion": "A conclusion sentence."}

    result = sentences_for_field("limitations", sections)

    assert result == [("First sentence here.", "discussion"), ("Second one too.", "discussion")]


def test_falls_through_to_the_next_target_section_when_the_first_is_absent() -> None:
    sections = {"conclusion": "A conclusion sentence."}

    result = sentences_for_field("limitations", sections)

    assert result == [("A conclusion sentence.", "conclusion")]


def test_empty_list_when_no_target_section_is_present() -> None:
    assert sentences_for_field("limitations", {"references": "Some citation text."}) == []


def test_empty_list_for_a_field_with_no_field_sections_entry() -> None:
    assert sentences_for_field("problem", {"introduction": "Some text."}) == []


def test_empty_list_when_sections_is_empty() -> None:
    assert sentences_for_field("method", {}) == []


def test_skips_a_present_but_blank_section_and_falls_through() -> None:
    sections = {"discussion": "   ", "limitations": "Real limitation text."}

    result = sentences_for_field("limitations", sections)

    assert result == [("Real limitation text.", "limitations")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_sections.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.extraction.sections'`

- [ ] **Step 3: Implement**

```python
# src/researchbridge/extraction/sections.py
"""Section-aware sentence lookup shared by HeuristicExtractor and
SemanticExtractor (Sec 46 full-text-aware extraction).
"""

from __future__ import annotations

from researchbridge.extraction.base import FIELD_SECTIONS
from researchbridge.extraction.sentences import split_sentences


def sentences_for_field(field: str, sections: dict[str, str]) -> list[tuple[str, str]]:
    """Returns [(sentence, section_name), ...] for the first of `field`'s
    FIELD_SECTIONS entries that's present (and non-blank) in `sections` -
    section priority, not a merged pool: sentences are drawn from exactly
    one section per call, never combined across the tuple. Empty list if
    `field` has no FIELD_SECTIONS entry, or none of its target sections
    are present - the caller's cue to fall back to the abstract.
    """
    for section_name in FIELD_SECTIONS.get(field, ()):
        text = sections.get(section_name)
        if text and text.strip():
            return [(sentence, section_name) for sentence in split_sentences(text)]
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_sections.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/sections.py tests/test_extraction_sections.py
git commit -m "feat(extraction): add sentences_for_field section-aware lookup"
```

---

### Task 3: `extraction/heuristic.py` - full-text-aware cue-phrase matching

**Files:**
- Modify: `src/researchbridge/extraction/heuristic.py`
- Modify: `tests/test_extraction_heuristic.py`

**Interfaces:**
- Consumes: `sentences_for_field` (Task 2), `ClaimCandidate` (Task 1).
- Produces: `HeuristicExtractor.extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]`.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_extraction_heuristic.py`:

```python
from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.extraction.heuristic import HeuristicExtractor, split_sentences


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def test_split_sentences_handles_basic_punctuation() -> None:
    assert split_sentences("We propose X. It works well! Does it scale?") == [
        "We propose X.", "It works well!", "Does it scale?",
    ]


def test_empty_abstract_yields_no_candidates() -> None:
    assert HeuristicExtractor().extract(_paper(None), {}) == []
    assert HeuristicExtractor().extract(_paper("   "), {}) == []


def test_method_cue_phrase_is_matched() -> None:
    abstract = "Prior work struggles with X. We propose a graph-based method for X. It works on real data."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We propose a graph-based method for X."
    assert method.confidence == "medium"
    assert method.section == "abstract"


def test_every_candidate_is_a_verbatim_substring_of_the_abstract() -> None:
    abstract = (
        "Existing systems fail under load. We ask whether a simpler design suffices. "
        "We propose a lock-free queue. Our results show a 3x speedup. "
        "However, throughput degrades past 64 threads."
    )
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_field_with_no_cue_match_is_omitted_not_fabricated() -> None:
    # no cue phrase for any field except an implicit "problem" opening sentence
    abstract = "Something happens in this domain."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    types = {c.claim_type for c in candidates}
    assert "dataset" not in types
    assert "applications" not in types


def test_problem_falls_back_to_first_sentence_at_low_confidence() -> None:
    abstract = "Coordination across regions is expensive. We propose a cheaper protocol."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    problem = next(c for c in candidates if c.claim_type == "problem")
    assert problem.claim_text == "Coordination across regions is expensive."
    assert problem.confidence == "low"


def test_problem_not_added_when_a_real_cue_phrase_already_covers_it() -> None:
    # first sentence itself matches a stronger cue phrase (method) -> no
    # separate low-confidence "problem" duplicate of the same sentence
    abstract = "We propose a new caching layer. It reduces latency significantly."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    problems = [c for c in candidates if c.claim_type == "problem"]
    assert problems == []


def test_research_gap_cue_phrase_is_matched() -> None:
    abstract = "We propose a new caching layer. Extending this to distributed settings remains future work."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    gap = next(c for c in candidates if c.claim_type == "research_gap")
    assert gap.claim_text == "Extending this to distributed settings remains future work."
    assert gap.confidence == "medium"


def test_future_work_no_longer_counted_as_a_limitation() -> None:
    # "future work" used to sit in the limitations cue list, conflating "a
    # weakness of this work" with "what's left for next time" - Sec 32
    # treats them as distinct explicit-gap categories.
    abstract = "We propose a new caching layer. Extending this to distributed settings remains future work."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    types = {c.claim_type for c in candidates}
    assert "limitations" not in types
    assert "research_gap" in types


def test_more_specific_phrase_takes_priority_over_vaguer_one() -> None:
    # "our contribution" (main_contribution) should win over a later, vaguer
    # "we show that" sentence also present in the same abstract
    abstract = "Our contribution is a new proof technique. Separately, we show that it generalizes."
    candidates = HeuristicExtractor().extract(_paper(abstract), {})

    contribution = next(c for c in candidates if c.claim_type == "main_contribution")
    assert contribution.claim_text == "Our contribution is a new proof technique."


def test_full_text_match_wins_over_an_abstract_match_for_the_same_field() -> None:
    abstract = "However, the abstract only vaguely gestures at a limitation."
    sections = {"discussion": "However, throughput degrades past 64 threads under load."}
    candidates = HeuristicExtractor().extract(_paper(abstract), sections)

    limitations = [c for c in candidates if c.claim_type == "limitations"]
    assert len(limitations) == 1
    assert limitations[0].claim_text == "However, throughput degrades past 64 threads under load."


def test_full_text_cue_phrase_hit_gets_high_confidence() -> None:
    abstract = "An abstract with no limitation cue phrase at all."
    sections = {"discussion": "However, throughput degrades past 64 threads under load."}
    candidates = HeuristicExtractor().extract(_paper(abstract), sections)

    limitation = next(c for c in candidates if c.claim_type == "limitations")
    assert limitation.confidence == "high"
    assert limitation.section == "discussion"


def test_full_text_evidence_quote_is_grounded_in_the_section_not_the_abstract() -> None:
    abstract = "A short abstract that does not mention the method at all."
    sections = {"methods": "We propose a graph-based method for X."}
    candidates = HeuristicExtractor().extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.evidence_quote == "We propose a graph-based method for X."
    assert method.evidence_quote not in abstract


def test_falls_back_to_abstract_when_full_text_section_has_no_cue_phrase_match() -> None:
    abstract = "We propose a graph-based method for X."
    sections = {"methods": "This section describes implementation details with no cue phrase at all."}
    candidates = HeuristicExtractor().extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.section == "abstract"
    assert method.confidence == "medium"


def test_problem_stays_abstract_only_even_with_full_text_available() -> None:
    # "problem" has no FIELD_SECTIONS entry by design - its fallback never
    # looks at full text, regardless of what sections are available
    abstract = "Coordination across regions is expensive."
    sections = {"introduction": "This paper is about coordination in distributed systems."}
    candidates = HeuristicExtractor().extract(_paper(abstract), sections)

    problem = next(c for c in candidates if c.claim_type == "problem")
    assert problem.section == "abstract"
    assert problem.claim_text == "Coordination across regions is expensive."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_heuristic.py -v`
Expected: FAIL - every test hits `TypeError: extract() missing 1 required positional argument: 'sections'`

- [ ] **Step 3: Implement**

Replace the full contents of `src/researchbridge/extraction/heuristic.py`:

```python
"""Baseline Extractor (Sec 28/29): cue-phrase sentence matching over the
abstract, or a full-text section when one exists and has a match (Sec 46).

Deliberately naive - the extraction analogue of the retrieval package's
TF-IDF/BM25 baselines (see retrieval/tfidf.py, bm25.py). The point of a
baseline is to be simple enough to establish a floor that a real method
(a local LLM, later) has to beat, not to be a good extractor. Precision
here is expected to be mediocre; that's the honest starting number Sec 28's
evaluation is meant to produce, not a defect to paper over.

For each Sec 28 field, first try sentences_for_field(field, sections) (the
paper's own most-likely full-text section for that field, per
FIELD_SECTIONS priority order) - a cue-phrase hit there gets
confidence="high" (Sec 15: "explicit statement in a named section"). If
that's empty (no full text, or no section recognized), fall back to
searching paper.abstract exactly as before Sec 46, confidence="medium". No
cue match anywhere -> no candidate for that field, never a fabricated
guess. "problem" is the one exception: most abstracts open by stating the
problem/motivation before any of the other cue phrases appear, so it falls
back to the first abstract sentence at low confidence when no stronger cue
phrase matches - this fallback is abstract-only by design (see
FIELD_SECTIONS's docstring), full text is never consulted for it.

Every candidate's evidence_quote is a verbatim sentence from wherever it
was matched, so it is grounded by construction - the pipeline's grounding
check (extraction/pipeline.py::_quote_is_grounded) will always pass.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.sections import sentences_for_field
from researchbridge.extraction.sentences import split_sentences

HEURISTIC_MODEL_VERSION = "cue-phrase-v1"

# Ordered by specificity: first matching phrase wins per field, so a more
# distinctive phrase (e.g. "our contribution") should sit before a vaguer
# one (e.g. "we show") that could just as easily belong to another field.
_CUE_PHRASES: dict[str, list[str]] = {
    "research_question": [
        "we ask whether", "we investigate whether", "this raises the question",
        "we study whether", "we study how", "the question of", "whether it is possible",
    ],
    "method": [
        "we propose", "we present", "we introduce", "our approach", "our method",
        "we develop", "we design", "we build",
    ],
    "main_contribution": [
        "our contribution", "our main contribution", "to the best of our knowledge",
        "for the first time", "we are the first", "we show that", "we demonstrate that",
    ],
    "dataset": [
        "we collect", "we release", "we construct a dataset", "trained on", "evaluated on",
        "benchmark dataset", "our dataset",
    ],
    "results": [
        "results show", "our results", "experiments show", "experimental results",
        "we achieve", "outperforms", "improves over",
    ],
    "limitations": [
        "however,", "a limitation", "does not", "fails to", "remains challenging",
        "is limited to",
    ],
    "applications": [
        "can be applied to", "is applicable to", "useful for", "in applications such as",
        "real-world applications",
    ],
    # Sec 32's "explicit gaps": a gap the paper states remains open, distinct
    # from limitations (a weakness of the current work) - "future work" used
    # to sit under limitations, which conflated "what's wrong with this"
    # with "what's left to do next".
    "research_gap": [
        "future work", "in future work", "we leave", "remains an open",
        "remains open", "an open question", "open problem", "yet to be explored",
        "we plan to", "future research", "further exploration",
    ],
}

class HeuristicExtractor:
    extraction_method = "heuristic"
    model_version = HEURISTIC_MODEL_VERSION

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        abstract_sentences = (
            split_sentences(paper.abstract) if paper.abstract and paper.abstract.strip() else []
        )

        candidates: list[ClaimCandidate] = []
        used_abstract_sentences: set[str] = set()

        for field, phrases in _CUE_PHRASES.items():
            full_text_pairs = sentences_for_field(field, sections)
            if full_text_pairs:
                match = _first_matching_pair(full_text_pairs, phrases)
                if match is not None:
                    sentence, section_name = match
                    candidates.append(
                        ClaimCandidate(field, sentence, sentence, confidence="high", section=section_name)
                    )
                    continue  # full text produced a candidate - it wins over the abstract (Sec 46)

            sentence = _first_matching_sentence(abstract_sentences, phrases)
            if sentence is not None:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence="medium"))
                used_abstract_sentences.add(sentence)

        # skip the fallback if the opening sentence was already claimed by a
        # real cue-phrase match above - relabeling it "problem" too would be
        # a duplicate, and quite possibly a mislabel (a method sentence isn't
        # the problem statement just because it happens to open the abstract)
        if abstract_sentences and abstract_sentences[0] not in used_abstract_sentences:
            candidates.append(
                ClaimCandidate("problem", abstract_sentences[0], abstract_sentences[0], confidence="low")
            )

        return candidates


def _first_matching_sentence(sentences: list[str], phrases: list[str]) -> str | None:
    for phrase in phrases:
        for sentence in sentences:
            if phrase in sentence.lower():
                return sentence
    return None


def _first_matching_pair(pairs: list[tuple[str, str]], phrases: list[str]) -> tuple[str, str] | None:
    for phrase in phrases:
        for sentence, section_name in pairs:
            if phrase in sentence.lower():
                return sentence, section_name
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_heuristic.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/heuristic.py tests/test_extraction_heuristic.py
git commit -m "feat(extraction): wire HeuristicExtractor to full-text sections"
```

---

### Task 4: `extraction/semantic.py` - full-text-aware similarity matching

**Files:**
- Modify: `src/researchbridge/extraction/semantic.py`
- Modify: `tests/test_extraction_semantic.py`

**Interfaces:**
- Consumes: `sentences_for_field` (Task 2), `ClaimCandidate` (Task 1).
- Produces: `SemanticExtractor.extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]`.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_extraction_semantic.py`:

```python
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import Paper
from researchbridge.extraction import semantic as semantic_module
from researchbridge.extraction.semantic import SemanticExtractor

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Bag-of-words cosine similarity - deterministic and legible, unlike a
    hash-based fake, so tests can reason directly about which sentence
    should win for a given field query instead of trusting opaque luck."""

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


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def _use_controlled_field_queries(monkeypatch, queries: dict[str, str]) -> None:
    """Swaps the production field-query prose for short controlled text, so
    tests reason about overlap counts they chose, not incidental word
    overlap in whatever the real prompts happen to say."""
    monkeypatch.setattr(semantic_module, "_FIELD_QUERIES", queries)


def test_empty_abstract_yields_no_candidates() -> None:
    extractor = SemanticExtractor(WordOverlapEmbedder())
    assert extractor.extract(_paper(None), {}) == []
    assert extractor.extract(_paper("   "), {}) == []


def test_picks_the_more_similar_sentence(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task. Weather today is unrelated and fine."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We use a graph neural network approach for this task."
    assert method.section == "abstract"


def test_every_candidate_is_a_verbatim_substring_of_the_abstract(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load system", "method": "graph neural network approach"},
    )
    abstract = "Latency under load breaks the system. We use a graph neural network approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_zero_overlap_field_gets_no_candidate(monkeypatch) -> None:
    # the field query shares no vocabulary at all with any abstract sentence
    # -> similarity 0.0, well under MIN_SIMILARITY -> omitted, not guessed
    _use_controlled_field_queries(monkeypatch, {"applications": "zzqx wwky unrelated_token_xyz"})
    abstract = "We use a graph neural network approach for this task."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    assert candidates == []


def test_strong_overlap_gets_medium_confidence(monkeypatch) -> None:
    # 2 of 2 field-query words present, diluted by the sentence's other 2
    # words -> cosine = 2/sqrt(2*4) ~= 0.707, above MEDIUM_CONFIDENCE_SIMILARITY (0.28)
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural"})
    abstract = "Graph neural nets work."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    expected_similarity = 2 / math.sqrt(2 * 4)
    assert expected_similarity >= semantic_module.MEDIUM_CONFIDENCE_SIMILARITY
    assert candidates[0].confidence == "medium"


def test_weak_but_above_threshold_overlap_gets_low_confidence(monkeypatch) -> None:
    assert semantic_module.MIN_SIMILARITY < semantic_module.MEDIUM_CONFIDENCE_SIMILARITY

    # 1 shared word ("graph") out of 3 field-query words and 6 sentence
    # words -> cosine = 1/sqrt(3*6) ~= 0.236, between MIN_SIMILARITY (0.15)
    # and MEDIUM_CONFIDENCE_SIMILARITY (0.28) - verified directly, not just
    # asserted, since bag-of-words cosine dilutes fast and isn't obvious
    # from word counts alone.
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network"})
    abstract = "Graph theory helps model networks well."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    expected_similarity = 1 / math.sqrt(3 * 6)
    assert semantic_module.MIN_SIMILARITY <= expected_similarity < semantic_module.MEDIUM_CONFIDENCE_SIMILARITY
    assert len(candidates) == 1
    assert candidates[0].confidence == "low"


def test_one_sentence_cannot_win_two_fields(monkeypatch) -> None:
    # Reproduces a real reported bug: a generic sentence that scores
    # decently against several field anchors used to get duplicated across
    # all of them, crowding out a more specific sentence that was each
    # field's true best match but nobody's *own* best match. Here "generic"
    # is the top choice for both "problem" and "dataset", but scores higher
    # for "problem" - so "dataset" must fall back to its only remaining
    # suitor, "specific", rather than reuse "generic".
    _use_controlled_field_queries(
        monkeypatch,
        {
            "problem": "graph neural network approach data challenge",
            "dataset": "graph neural network data",
        },
    )
    abstract = "Generic graph neural network approach data challenge. Specific graph data only."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    by_field = {c.claim_type: c.claim_text for c in candidates}
    assert by_field["problem"].startswith("Generic")
    assert by_field["dataset"].startswith("Specific")
    assert by_field["problem"] != by_field["dataset"]


def test_fields_are_evaluated_independently(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load", "dataset": "zzqx wwky unrelated_token_xyz"},
    )
    abstract = "Latency under load breaks the system. We use a graph approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    types = {c.claim_type for c in candidates}
    assert "problem" in types
    assert "dataset" not in types


def test_full_text_match_wins_over_an_abstract_match_for_the_same_field(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task."
    sections = {"methods": "We use a graph neural network approach in the methods section specifically."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.section == "methods"
    assert "methods section specifically" in method.claim_text


def test_full_text_confidence_is_not_bumped_above_the_similarity_banding(monkeypatch) -> None:
    # unlike HeuristicExtractor, a semantic full-text hit keeps the same
    # medium/low confidence banding as an abstract hit would - similarity
    # scores aren't the same kind of certainty as a verbatim cue-phrase hit
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural"})
    sections = {"methods": "Graph neural nets work."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(None), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.confidence == "medium"  # same banding rule as the abstract-only test above
    assert method.section == "methods"


def test_two_fields_sharing_a_target_section_still_cannot_win_the_same_sentence(monkeypatch) -> None:
    # method and dataset both target ("methods", "experiments") in
    # FIELD_SECTIONS - this reproduces test_one_sentence_cannot_win_two_fields
    # but for two fields sharing a resolved full-text section, proving the
    # collision-avoidance grouping (not just the abstract-only pool) works
    _use_controlled_field_queries(
        monkeypatch,
        {
            "method": "graph neural network approach data challenge",
            "dataset": "graph neural network data",
        },
    )
    sections = {"methods": "Generic graph neural network approach data challenge. Specific graph data only."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(None), sections)

    by_field = {c.claim_type: c.claim_text for c in candidates}
    assert by_field["method"].startswith("Generic")
    assert by_field["dataset"].startswith("Specific")
    assert by_field["method"] != by_field["dataset"]


def test_falls_back_to_abstract_when_full_text_section_has_zero_overlap(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task."
    sections = {"methods": "zzqx wwky unrelated_token_xyz completely different content here."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.section == "abstract"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_semantic.py -v`
Expected: FAIL - every test hits `TypeError: extract() missing 1 required positional argument: 'sections'`

- [ ] **Step 3: Implement**

Replace the full contents of `src/researchbridge/extraction/semantic.py`:

```python
"""Improvement over the heuristic baseline (Sec 26): embedding-similarity
sentence selection instead of fixed cue phrases, over the abstract or a
full-text section when one exists (Sec 46).

Still extractive, not generative - for each Sec 28 field, embed every
candidate sentence and pick whichever is closest to a description of that
field, taking the sentence itself as the claim. That keeps the same
verbatim-grounding guarantee the heuristic extractor has (the pipeline's
grounding check in extraction/pipeline.py requires an exact substring of
whatever section produced it), which a generative model answering in its
own words could not offer without weakening Sec 15's "no Grounding
Illusion" rule - not a call to make quietly while chasing a better score.

The heuristic baseline's weakest fields (research_question, applications:
0 recall) are exactly where fixed strings are the wrong tool - real
abstracts phrase a research question or an application in enormously
varied ways, and no small phrase list will catch most of them. Semantic
similarity doesn't need the wording to match, only the meaning to.

Uses the Week 6 embedding model (SentenceTransformerEmbedder / all-
MiniLM-L6-v2) - no new model, no new dependency, and it's already fast
enough for this: one batched embed_texts() call per candidate pool covers
every sentence and every field description in that pool at once.

Each sentence proposes only its own single best-matching field WITHIN ITS
OWN CANDIDATE POOL (a real bug this fixed: on short, jargon-dense
abstracts, all-MiniLM-L6-v2 can rate a fluent-but-content-free sentence -
e.g. "This report has been updated for posterity..." - higher against
several field anchors than the sentence that actually describes the
method, because it rewards grammatically ordinary prose over jargon-heavy
technical prose almost independent of topical relevance. Picking each
field's best sentence independently let that one attractive-but-wrong
sentence get duplicated across four different fields on a real paper.
Requiring each sentence to serve at most one field - whichever it matches
best - means a field with no sentence that actually prefers it abstains
instead of inheriting someone else's leftover.

"Within its own candidate pool" matters once full text is involved: two
fields that resolve to the SAME full-text section (e.g. method and
dataset both -> methods/experiments) share a real collision risk and must
be matched together; two fields resolving to different sections (or one
resolving to a section and another falling back to the abstract) have
disjoint sentence pools and can never collide with each other. extract()
groups fields by their exact resolved pool before running the matching -
see _match_pool.
"""

from __future__ import annotations

from collections import defaultdict

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.sections import sentences_for_field
from researchbridge.extraction.sentences import split_sentences

SEMANTIC_MODEL_VERSION = "semantic-v3"

# One anchor description per Sec 28 field - the sentence most similar to
# this description (by cosine similarity, since Embedder vectors are
# L2-normalized) becomes that field's candidate.
_FIELD_QUERIES: dict[str, str] = {
    "problem": "The problem or challenge that motivates this research.",
    "research_question": "The research question this paper investigates.",
    "method": "The method, approach, or technique this paper proposes.",
    "dataset": "The dataset or data source used in this work.",
    "main_contribution": "The main contribution or novel finding of this paper.",
    "results": "The results or findings reported in this paper.",
    "limitations": "A limitation or weakness acknowledged in this paper.",
    "applications": "A real-world application this work supports.",
    "research_gap": "A gap or open question the paper says remains for future work.",
}

# Calibrated against this model's actual score distribution, not guessed:
# all-MiniLM-L6-v2 scores a short instructional anchor ("The problem or
# challenge...") much lower against real sentence prose than intuition
# suggests - measured across the real 40-paper benchmark, best-match
# similarity per (paper, field) runs from 0.02 to 0.58, median 0.22, with
# only 10% clearing 0.34. A first attempt at 0.30 sat near the 90th
# percentile and made the extractor abstain on nearly everything. These
# numbers come from the raw similarity distribution alone, never from
# ground-truth match/no-match labels - reading the instrument's own scale,
# not tuning a threshold against the benchmark it's evaluated on. Reused
# unchanged for full-text sentences (Sec 46) - re-tuning for that
# distribution is future work once real full-text extraction data exists.
MIN_SIMILARITY = 0.15
MEDIUM_CONFIDENCE_SIMILARITY = 0.28


class SemanticExtractor:
    extraction_method = "semantic"
    model_version = SEMANTIC_MODEL_VERSION

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        abstract_sentences = (
            split_sentences(paper.abstract) if paper.abstract and paper.abstract.strip() else []
        )

        # Group fields by the EXACT pool they'll draw from - same pool
        # means real collision risk, different pools can never collide.
        pool_fields: dict[str, list[str]] = defaultdict(list)
        pool_sentences: dict[str, list[str]] = {}
        pool_section: dict[str, str | None] = {}

        for field in _FIELD_QUERIES:
            full_text_pairs = sentences_for_field(field, sections)
            if full_text_pairs:
                section_name = full_text_pairs[0][1]  # one sentences_for_field call = one section
                pool_key = f"fulltext:{section_name}"
                pool_sentences.setdefault(pool_key, [s for s, _ in full_text_pairs])
                pool_section[pool_key] = section_name
            else:
                pool_key = "abstract"
                pool_sentences.setdefault(pool_key, abstract_sentences)
                pool_section[pool_key] = None
            pool_fields[pool_key].append(field)

        candidates: list[ClaimCandidate] = []
        for pool_key, fields in pool_fields.items():
            candidates.extend(self._match_pool(fields, pool_sentences[pool_key], pool_section[pool_key]))
        return candidates

    def _match_pool(
        self, fields: list[str], sentences: list[str], section_name: str | None
    ) -> list[ClaimCandidate]:
        if not sentences:
            return []

        vectors = self.embedder.embed_texts(sentences + [_FIELD_QUERIES[f] for f in fields])
        sentence_vectors, field_vectors = vectors[: len(sentences)], vectors[len(sentences) :]

        # each sentence's own best-matching field (within this pool only),
        # and its score there
        proposals: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sentence, svec in zip(sentences, sentence_vectors, strict=True):
            similarities = {
                field: sum(a * b for a, b in zip(svec, fvec, strict=True))
                for field, fvec in zip(fields, field_vectors, strict=True)
            }
            best_field = max(similarities, key=similarities.get)
            proposals[best_field].append((sentence, similarities[best_field]))

        candidates: list[ClaimCandidate] = []
        for field in fields:
            suitors = proposals.get(field)
            if not suitors:
                continue
            sentence, similarity = max(suitors, key=lambda pair: pair[1])
            if similarity < MIN_SIMILARITY:
                continue
            confidence = "medium" if similarity >= MEDIUM_CONFIDENCE_SIMILARITY else "low"
            if section_name is not None:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence, section=section_name))
            else:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence))
        return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_semantic.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/semantic.py tests/test_extraction_semantic.py
git commit -m "feat(extraction): wire SemanticExtractor to full-text sections"
```

---

### Task 5: `extraction/hybrid.py` - thread sections through, fix confidence routing

**Files:**
- Modify: `src/researchbridge/extraction/hybrid.py`
- Modify: `tests/test_extraction_hybrid.py`

**Interfaces:**
- Consumes: `HeuristicExtractor.extract`/`SemanticExtractor.extract` (Tasks 3-4, both now `(paper, sections)`).
- Produces: `HybridExtractor.extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]`.

- [ ] **Step 1: Write the failing test**

Replace the full contents of `tests/test_extraction_hybrid.py`:

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass

from researchbridge.db.models import Paper
from researchbridge.extraction.hybrid import HybridExtractor


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


@dataclass
class FakeEmbedder:
    """Deterministic and simple, not realistic - these tests check routing
    (does the hybrid prefer the right sub-extractor's answer), not semantic
    quality, so any valid unit vectors are enough. HybridExtractor always
    runs both sub-extractors before routing (one batched embed_texts call
    is cheaper than deciding per-field whether it's needed), so every test
    here needs an embedder that actually works, not one that refuses."""

    model_name: str = "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if len(t) % 2 == 0 else [0.0, 1.0] for t in texts]


def test_empty_abstract_yields_no_candidates() -> None:
    extractor = HybridExtractor(FakeEmbedder())
    assert extractor.extract(_paper(None), {}) == []


def test_problem_always_prefers_heuristic_even_at_low_confidence() -> None:
    # heuristic's problem answer is its first-sentence fallback (low
    # confidence by construction) - the routing rule still prefers it for
    # "problem" specifically, regardless of what semantic also found
    abstract = "Coordination across regions is expensive."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), {})

    problem = next(c for c in candidates if c.claim_type == "problem")
    assert problem.claim_text == "Coordination across regions is expensive."
    assert problem.confidence == "low"  # heuristic's own confidence is preserved, not overridden


def test_medium_confidence_heuristic_hit_wins_over_semantic() -> None:
    # "we propose" is a real cue phrase (heuristic/method) -> medium
    # confidence -> routing must prefer it over whatever semantic found
    abstract = "We propose a lock-free queue for high throughput systems."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), {})

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We propose a lock-free queue for high throughput systems."
    assert method.confidence == "medium"


def test_field_with_no_heuristic_match_falls_back_to_semantic() -> None:
    # no cue phrase in this abstract matches any field's phrase list, so
    # every non-problem field (if present at all) must have come from semantic
    abstract = "Something happens in this domain. Another sentence follows it."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), {})

    non_problem = [c for c in candidates if c.claim_type != "problem"]
    assert all(c.claim_text is not None for c in non_problem)  # came from somewhere, not a crash
    # every non-problem candidate's confidence is semantic's own banding
    # (medium/low from cosine similarity), never heuristic's - since
    # heuristic had no cue-phrase match to contribute for any of these
    assert all(c.confidence in ("medium", "low") for c in non_problem)


def test_grounding_holds_for_every_hybrid_candidate() -> None:
    abstract = "Coordination across regions is expensive. We propose a cheaper protocol."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), {})

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_no_duplicate_fields_in_output() -> None:
    abstract = "We propose a cheaper protocol for coordination. It reduces overhead significantly."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), {})

    types = [c.claim_type for c in candidates]
    assert len(types) == len(set(types))


def test_high_confidence_full_text_heuristic_hit_still_wins_over_semantic() -> None:
    # a full-text cue-phrase hit is confidence="high" (Task 3), not
    # "medium" - _choose()'s routing rule must still prefer it over
    # semantic, the same way it already prefers a "medium" abstract hit
    abstract = "An abstract with no method cue phrase at all."
    sections = {"methods": "We propose a graph-based method for X."}
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We propose a graph-based method for X."
    assert method.confidence == "high"
    assert method.section == "methods"


def test_sections_are_threaded_through_to_both_sub_extractors() -> None:
    abstract = "An abstract with no limitations cue phrase at all."
    sections = {"discussion": "However, throughput degrades past 64 threads under load."}
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract), sections)

    limitations = next(c for c in candidates if c.claim_type == "limitations")
    assert limitations.section == "discussion"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_hybrid.py -v`
Expected: FAIL - every test hits `TypeError: extract() missing 1 required positional argument: 'sections'`

- [ ] **Step 3: Implement**

Replace the full contents of `src/researchbridge/extraction/hybrid.py`:

```python
"""Combines the heuristic and semantic extractors (Sec 26 Improvement step).

Per-field comparison against the benchmark showed the two baselines win on
almost entirely different fields: heuristic on problem/method/limitations,
semantic on research_question/main_contribution/results/applications (see
the commit that added semantic.py). It is tempting to just hardcode that
per-field routing - but that would mean choosing rules by reading the
score table off the exact 40 papers this extractor gets re-evaluated
against, which quietly launders an inflated number rather than measuring
anything. The routing below is deliberately NOT built that way.

Two rules, both justified without looking at the benchmark's ground truth:

1. A literal cue-phrase hit (heuristic confidence in "medium"/"high" - see
   below) is trusted over semantic similarity when one exists - an
   explicit, specific textual signal is stronger evidence than a general
   similarity score when both are available.

2. "problem" specifically always prefers the heuristic's answer, including
   its low-confidence first-sentence fallback. This is because abstracts
   conventionally open with the problem/motivation before anything else -
   a documented property of how scientific abstracts are written, not
   something read off this benchmark's answers.

Everywhere else, semantic fills in where heuristic found no cue-phrase
match at all - the situation semantic exists for (Sec 26): fields whose
real-world phrasing varies too much for a fixed phrase list.

Rule 1's confidence check covers both "medium" (an abstract-only cue-phrase
hit) and "high" (a full-text, named-section cue-phrase hit - Sec 46) -
both mean the same thing this rule cares about, "an explicit cue-phrase
match exists", so a full-text heuristic hit is preferred over semantic
exactly like an abstract one always was.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.heuristic import HeuristicExtractor
from researchbridge.extraction.semantic import SemanticExtractor

HYBRID_MODEL_VERSION = "hybrid-v1"


class HybridExtractor:
    extraction_method = "hybrid"
    model_version = HYBRID_MODEL_VERSION

    def __init__(self, embedder: Embedder) -> None:
        self._heuristic = HeuristicExtractor()
        self._semantic = SemanticExtractor(embedder)

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        heuristic_by_field = {c.claim_type: c for c in self._heuristic.extract(paper, sections)}
        semantic_by_field = {c.claim_type: c for c in self._semantic.extract(paper, sections)}

        fields = heuristic_by_field.keys() | semantic_by_field.keys()
        chosen = (
            self._choose(field, heuristic_by_field.get(field), semantic_by_field.get(field))
            for field in fields
        )
        return [c for c in chosen if c is not None]

    def _choose(
        self, field: str, heuristic: ClaimCandidate | None, semantic: ClaimCandidate | None
    ) -> ClaimCandidate | None:
        if field == "problem":
            return heuristic or semantic
        if heuristic is not None and heuristic.confidence in ("medium", "high"):
            return heuristic
        return semantic or heuristic
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_hybrid.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/hybrid.py tests/test_extraction_hybrid.py
git commit -m "feat(extraction): thread sections through HybridExtractor, fix confidence routing"
```

---

### Task 6: `extraction/pipeline.py` - look up full text, ground and persist per-section

**Files:**
- Modify: `src/researchbridge/extraction/pipeline.py`
- Modify: `tests/test_extraction_pipeline.py`

**Interfaces:**
- Consumes: `PaperFullText` (existing model, `db/models.py`), the updated `Extractor.extract(paper, sections)` (Tasks 3-5).
- Produces: no new public interface - `ExtractionPipeline` itself is unchanged from the outside.

- [ ] **Step 1: Write the failing test**

In `tests/test_extraction_pipeline.py`, update `FakeExtractor` and add new tests. First, replace the `FakeExtractor` class and the imports:

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from researchbridge.db.models import (
    CandidateGap,
    CandidateGapEvidence,
    Evidence,
    ExtractedClaim,
    ExtractionError,
    ExtractionRun,
    Paper,
    PaperFullText,
)
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.pipeline import ExtractionPipeline, reset_extraction_data
from researchbridge.extraction.stub import STUB_MODEL_VERSION, StubExtractor


def _make_paper(session, source_id: str, abstract: str | None = "A short abstract.") -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source="fake",
        source_id=source_id,
        title="A Paper",
        abstract=abstract,
        raw_metadata={},
        ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


@dataclass
class FakeExtractor:
    candidates_by_paper_id: dict = field(default_factory=dict)
    extraction_method: str = "fake"
    model_version: str = "fake-v1"
    raises_for: set = field(default_factory=set)
    sections_seen: dict = field(default_factory=dict)
    """paper_id -> the `sections` dict this extractor was actually called
    with, for tests that need to confirm the pipeline looked up and passed
    through the right paper_fulltext row."""

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        self.sections_seen[paper.id] = sections
        if paper.id in self.raises_for:
            raise RuntimeError("boom")
        return self.candidates_by_paper_id.get(paper.id, [])
```

`StubExtractor` (used by most existing tests, e.g. `test_stub_extractor_creates_grounded_evidence_and_claim`) needs its own signature updated too - see Task 6's implementation step for `stub.py`. Every existing test in this file that constructs `ExtractionPipeline(extractor=StubExtractor(), ...)` or `FakeExtractor(...)` stays unchanged (they don't call `.extract()` directly - the pipeline does), so no other edits are needed to the existing ~11 tests. Add these new tests at the end of the file:

```python
def test_paper_with_no_fulltext_row_gets_an_empty_sections_dict(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1")
    session.close()

    extractor = FakeExtractor()
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    pipeline.run()

    assert extractor.sections_seen[paper.id] == {}


def test_paper_with_fulltext_row_gets_its_sections_passed_through(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1")
    session.add(
        PaperFullText(
            paper_id=paper.id, sections={"methods": "We propose a graph-based method."},
            source_url="https://example.com/p1.pdf",
        )
    )
    session.commit()

    extractor = FakeExtractor()
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    pipeline.run()

    assert extractor.sections_seen[paper.id] == {"methods": "We propose a graph-based method."}


def test_full_text_candidate_is_grounded_against_its_own_section_not_the_abstract(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="This abstract does not contain the method sentence.")
    session.add(
        PaperFullText(
            paper_id=paper.id, sections={"methods": "We propose a graph-based method for X."},
            source_url="https://example.com/p1.pdf",
        )
    )
    session.commit()

    extractor = FakeExtractor(
        candidates_by_paper_id={
            paper.id: [
                ClaimCandidate(
                    claim_type="method", claim_text="We propose a graph-based method for X.",
                    evidence_quote="We propose a graph-based method for X.", confidence="high",
                    section="methods",
                )
            ]
        }
    )
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.claims_created == 1  # grounded against sections["methods"], not the abstract

        claim = session.execute(select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)).scalar_one()
        evidence = session.get(Evidence, claim.evidence_id)
        assert evidence.section == "methods"
        assert evidence.source_locator == "methods"
    finally:
        session.close()


def test_full_text_candidate_with_a_quote_not_in_its_claimed_section_is_rejected(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1")
    session.add(
        PaperFullText(
            paper_id=paper.id, sections={"methods": "We propose a graph-based method for X."},
            source_url="https://example.com/p1.pdf",
        )
    )
    session.commit()

    extractor = FakeExtractor(
        candidates_by_paper_id={
            paper.id: [
                ClaimCandidate(
                    claim_type="method", claim_text="fabricated text",
                    evidence_quote="This sentence does not appear in the methods section.",
                    confidence="high", section="methods",
                )
            ]
        }
    )
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.claims_created == 0
        assert run.candidates_rejected == 1
    finally:
        session.close()


def test_abstract_section_candidate_persists_section_abstract(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="We propose a new method for X.")
    session.commit()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    pipeline.run()

    session = session_factory()
    try:
        claim = session.execute(select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)).scalar_one()
        evidence = session.get(Evidence, claim.evidence_id)
        assert evidence.section == "abstract"
    finally:
        session.close()
```

Note: the last new test (`test_abstract_section_candidate_persists_section_abstract`) asserts `evidence.section == "abstract"` (lowercase) - this is a real, deliberate behavior change: today's pipeline hardcodes `section="Abstract"` (capitalized). Check whether any other code reads `Evidence.section` expecting the capitalized literal before this task's Step 3 - grep confirms `extraction/pipeline.py` is the only writer, and no reader anywhere in the codebase pattern-matches on the literal string `"Abstract"` (evidence.section is otherwise only ever displayed, e.g. in the frontend's evidence gutter, never compared against a hardcoded value) - safe to lowercase for consistency with every full-text section name already being lowercase (`fulltext/parse.py`'s section keys).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_pipeline.py -v`
Expected: FAIL - the four new tests fail (`sections_seen` not populated / `TypeError` from `StubExtractor.extract` still taking one argument), existing tests still pass since they go through the pipeline, not `.extract()` directly

- [ ] **Step 3: Implement**

First, update `src/researchbridge/extraction/stub.py` - find its `extract` method and change its signature to match:

```python
def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
```

(the body is unchanged - `StubExtractor` only ever reads `paper.abstract`, matching this task's explicit "StubExtractor is untouched" behavior constraint; only the signature needs to accept and ignore the new parameter).

Then replace the full contents of `src/researchbridge/extraction/pipeline.py`:

```python
"""Extraction pipeline: select unprocessed papers -> extract -> verify -> persist.

Idempotent: only processes papers with no existing extracted_claims row, so
re-running is safe and (once a costlier extractor is ever used) won't
reprocess/re-bill already-processed papers. Every candidate passes through
two independent gates before anything is persisted, and failing either one
rejects the candidate and logs it to extraction_errors rather than silently
storing or silently relabeling it:

1. Grounding (_quote_is_grounded): the evidence_quote is verified against
   whatever section it claims to come from - paper.abstract when
   candidate.section == "abstract", or paper_fulltext.sections[candidate.
   section] when it's a real full-text section (Sec 46) - the concrete
   mechanism behind the blueprint's "no Grounding Illusion" rule, §15.
   This proves the quote is not fabricated.

2. Claim-type validation (extraction/validation.py::validate_claim_type):
   grounding alone does not prove the quote actually expresses the claim
   type it was filed under - a quote can be a verbatim, grounded substring
   of its section and still be a results sentence mislabeled as a
   research_gap. This is the semantic check for that.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import (
    CandidateGapEvidence,
    Evidence,
    ExtractedClaim,
    ExtractionError,
    ExtractionRun,
    Paper,
    PaperFullText,
    ResearchAssessmentEvidence,
)
from researchbridge.extraction.base import Extractor
from researchbridge.extraction.validation import validate_claim_type

logger = logging.getLogger(__name__)


def reset_extraction_data(session: Session) -> int:
    """Delete every prior extraction result so the next run reprocesses the
    whole corpus from scratch (the "force re-extract" path).

    Evidence rows can be referenced by CandidateGapEvidence and
    ResearchAssessmentEvidence (gap detection and assessments both cite
    specific evidence quotes) - those referencing rows have to go first or
    the Evidence delete below violates their foreign key. This does mean a
    force re-extract quietly drops support for any candidate gap or
    assessment that cited the old evidence; there's no cheap way to
    recompute those from the new extraction, so this is a deliberate
    "start over" operation, not a safe background refresh.

    Returns the number of Evidence rows removed (the corpus-wide count of
    what got reset).
    """
    session.execute(
        delete(CandidateGapEvidence).where(CandidateGapEvidence.evidence_id.in_(select(Evidence.id)))
    )
    session.execute(
        delete(ResearchAssessmentEvidence).where(
            ResearchAssessmentEvidence.evidence_id.in_(select(Evidence.id))
        )
    )
    session.execute(delete(ExtractedClaim))
    result = session.execute(delete(Evidence))
    session.execute(delete(ExtractionError))
    session.commit()
    return result.rowcount


class ExtractionPipeline:
    def __init__(self, extractor: Extractor, session_factory: sessionmaker[Session]) -> None:
        self.extractor = extractor
        self.session_factory = session_factory

    def run(self, limit: int | None = None, force: bool = False) -> str:
        session = self.session_factory()
        run = ExtractionRun(extractor_name=self.extractor.extraction_method, status="running", force=force)
        session.add(run)
        session.commit()
        run_id = str(run.id)

        try:
            papers = self._select_unprocessed_papers(session, limit)
            total = len(papers)
            logger.info(
                "Extraction run %s starting: %d paper(s) to process (extractor=%s)",
                run_id,
                total,
                self.extractor.extraction_method,
            )

            for i, paper in enumerate(papers, start=1):
                self._process_paper(session, run, paper)
                run.papers_processed += 1
                session.commit()
                if i % 10 == 0 or i == total:
                    logger.info(
                        "Extraction run %s: %d/%d papers processed (claims=%d, rejected=%d)",
                        run_id,
                        i,
                        total,
                        run.claims_created,
                        run.candidates_rejected,
                    )

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            session.commit()
            logger.info(
                "Extraction run %s completed: %d claim(s) created, %d rejected",
                run_id,
                run.claims_created,
                run.candidates_rejected,
            )

        except Exception as exc:  # noqa: BLE001 - run must record failure, not crash silently
            logger.exception("Extraction run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id

    def _process_paper(self, session: Session, run: ExtractionRun, paper: Paper) -> None:
        fulltext_row = session.execute(
            select(PaperFullText).where(PaperFullText.paper_id == paper.id)
        ).scalar_one_or_none()
        sections = fulltext_row.sections if fulltext_row is not None else {}

        try:
            candidates = self.extractor.extract(paper, sections)
        except Exception as exc:  # noqa: BLE001 - one extractor failure must not crash the run
            logger.exception("Extractor failed for paper %s", paper.id)
            self._record_error(session, run.id, paper.id, "extractor_error", str(exc)[:2000])
            return

        for candidate in candidates:
            haystack = paper.abstract if candidate.section == "abstract" else sections.get(candidate.section)
            if not _quote_is_grounded(candidate.evidence_quote, haystack):
                self._record_error(
                    session,
                    run.id,
                    paper.id,
                    "ungrounded_quote",
                    f"evidence_quote not found in section {candidate.section!r}: {candidate.evidence_quote!r}",
                )
                run.candidates_rejected += 1
                continue

            type_validation = validate_claim_type(candidate.claim_type, candidate.claim_text)
            if not type_validation.is_valid:
                self._record_error(
                    session,
                    run.id,
                    paper.id,
                    "semantic_type_mismatch",
                    f"claim_type={candidate.claim_type!r} rejected: {type_validation.reason} "
                    f"(text: {candidate.claim_text!r})",
                )
                run.candidates_rejected += 1
                continue

            try:
                with session.begin_nested():
                    evidence = Evidence(
                        paper_id=paper.id,
                        evidence_type=candidate.claim_type,
                        section=candidate.section,
                        text=candidate.evidence_quote,
                        source_locator=candidate.section,
                        extraction_method=self.extractor.extraction_method,
                        model_version=self.extractor.model_version,
                        confidence=candidate.confidence,
                    )
                    session.add(evidence)
                    session.flush()  # populate evidence.id for the FK below

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
                run.claims_created += 1
            except IntegrityError as exc:
                self._record_error(session, run.id, paper.id, "persist_error", str(exc)[:2000])

    def _select_unprocessed_papers(self, session: Session, limit: int | None) -> list[Paper]:
        already_extracted = select(ExtractedClaim.paper_id).distinct()
        query = select(Paper).where(Paper.id.notin_(already_extracted))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())

    def _record_error(self, session: Session, run_id: Any, paper_id: Any, error_type: str, detail: str) -> None:
        session.add(
            ExtractionError(
                extraction_run_id=run_id,
                paper_id=paper_id,
                error_type=error_type,
                error_detail=detail,
            )
        )


def _quote_is_grounded(quote: str, haystack: str | None) -> bool:
    if not quote or not haystack:
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(haystack)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_pipeline.py -v`
Expected: PASS (all tests, existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/extraction/pipeline.py src/researchbridge/extraction/stub.py tests/test_extraction_pipeline.py
git commit -m "feat(extraction): look up paper_fulltext, ground/persist per-section"
```

---

### Task 7: `extraction/cli_evaluate.py` - pass full text through to benchmark evaluation

**Files:**
- Modify: `src/researchbridge/extraction/cli_evaluate.py`
- Modify: `tests/test_extraction_cli_evaluate.py`

**Interfaces:**
- Consumes: `PaperFullText`, updated `Extractor.extract(paper, sections)`.
- Produces: no new public interface - `_evaluate_one` stays the internal function it already is.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_extraction_cli_evaluate.py` (check its existing imports first - it currently imports only `_build_results_json`/`write_results_json`/`FieldScore`; add the ones below alongside them):

```python
import uuid
from unittest.mock import Mock

from researchbridge.benchmark.store import Annotation
from researchbridge.db.models import Paper, PaperFullText
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.cli_evaluate import _evaluate_one


def _paper(source_id: str, abstract: str = "An abstract.") -> Paper:
    return Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def _annotation(source_id: str) -> Annotation:
    return Annotation(
        source_id=source_id, title="t", domain="ML", year=2024,
        fields={
            "problem": "", "research_question": "", "method": "", "dataset": "",
            "main_contribution": "", "results": "", "limitations": "", "applications": "",
        },
        research_gap={"addressed": "", "remaining": ""}, key_evidence=[],
    )


def test_evaluate_one_passes_empty_sections_when_paper_has_no_fulltext_row(session_factory) -> None:
    session = session_factory()
    paper = _paper("P1")
    session.add(paper)
    session.commit()

    extractor = Mock()
    extractor.extract.return_value = []
    annotation = _annotation("P1")

    _evaluate_one(session, extractor, [annotation], {"P1": paper}, embedder=Mock(), threshold=0.5)

    extractor.extract.assert_called_once_with(paper, {})
    session.close()


def test_evaluate_one_passes_a_papers_own_sections_when_a_fulltext_row_exists(session_factory) -> None:
    session = session_factory()
    paper = _paper("P1")
    session.add(paper)
    session.flush()
    session.add(
        PaperFullText(
            paper_id=paper.id, sections={"methods": "We propose a method."},
            source_url="https://example.com/p1.pdf",
        )
    )
    session.commit()

    extractor = Mock()
    extractor.extract.return_value = []
    annotation = _annotation("P1")

    _evaluate_one(session, extractor, [annotation], {"P1": paper}, embedder=Mock(), threshold=0.5)

    extractor.extract.assert_called_once_with(paper, {"methods": "We propose a method."})
    session.close()
```

Note: check `researchbridge.benchmark.store.Annotation`'s actual field names before running this (`tests/test_extraction_cli_evaluate.py`'s existing tests or `benchmark/store.py` itself will show the real dataclass shape) - adjust the `_annotation` helper above to match if it differs from what's shown here.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_cli_evaluate.py -v`
Expected: FAIL - `_evaluate_one` doesn't accept a `session` parameter yet (`TypeError`), and doesn't exist with that exact signature until Step 3

- [ ] **Step 3: Implement**

In `src/researchbridge/extraction/cli_evaluate.py`, add `PaperFullText` to the model import and `Session` to the sqlalchemy.orm import:

```python
from sqlalchemy import select
from sqlalchemy.orm import Session
```

```python
from researchbridge.db.models import Paper, PaperFullText
```

Replace `_evaluate_one` and its call site in `main()`:

```python
def _evaluate_one(
    session: Session, extractor, annotations, papers_by_source_id, embedder: Embedder, threshold: float
) -> dict[str, FieldScore]:
    paper_ids = [p.id for p in papers_by_source_id.values()]
    fulltext_by_paper_id = {
        row.paper_id: row.sections
        for row in session.execute(select(PaperFullText).where(PaperFullText.paper_id.in_(paper_ids))).scalars()
    }
    predictions = {
        a.source_id: extractor.extract(
            papers_by_source_id[a.source_id], fulltext_by_paper_id.get(papers_by_source_id[a.source_id].id, {})
        )
        for a in annotations
    }
    ground_truth = {
        a.source_id: {**a.fields, "research_gap": a.research_gap.get("remaining", "")} for a in annotations
    }
    fields = [*ANNOTATION_FIELDS, "research_gap"]
    return evaluate(predictions, ground_truth, fields, embedder, threshold)
```

And in `main()`, update the call site:

```python
            scores = _evaluate_one(session, extractor, usable, papers_by_source_id, embedder, args.threshold)
```

(was: `scores = _evaluate_one(extractor, usable, papers_by_source_id, embedder, args.threshold)` - just adds `session` as the first argument).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_cli_evaluate.py -v`
Expected: PASS (all existing tests + 2 new)

- [ ] **Step 5: Run the full extraction test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_extraction_base.py tests/test_extraction_sections.py tests/test_extraction_heuristic.py tests/test_extraction_semantic.py tests/test_extraction_hybrid.py tests/test_extraction_pipeline.py tests/test_extraction_cli_evaluate.py tests/test_extraction_validation.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/extraction/cli_evaluate.py tests/test_extraction_cli_evaluate.py
git commit -m "feat(extraction): pass paper_fulltext through to benchmark evaluation"
```

---

## Explicitly not in this plan

- Re-extracting the ~47,000 already-processed papers - this plan only changes behavior for new extraction runs (new papers, or an explicit `--force` re-extract).
- Re-running the Sec 28 benchmark evaluation and comparing before/after scores - Task 7 makes evaluation full-text-capable; actually re-running it is separate, later work once enough benchmark papers have full text.
- Re-tuning `MIN_SIMILARITY`/`MEDIUM_CONFIDENCE_SIMILARITY` for full-text sentence distributions (spec's "Open questions").
- Any change to `fulltext/`, `analysis_claims`, or `assessment/*` - this plan only extends what `extraction/` reads.
