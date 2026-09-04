# Full-Text-Aware Extraction — Design Spec

Source of truth for the gap this closes: `fulltext/` (shipped earlier this session) fetches and
stores each open-access paper's full text into `paper_fulltext.sections`, but nothing reads that
table. Every `Extractor` implementation (`HeuristicExtractor`, `SemanticExtractor`, `HybridExtractor`)
still only ever sees `paper.abstract`. This spec wires the extractors to actually use full text when
it exists, closing the gap flagged in `ResearchBridge.md`'s Current Implementation Status.

## Why this is architectural, not a small tweak

`Extractor.extract(self, paper: Paper) -> list[ClaimCandidate]` is a `Protocol` with three real
implementations plus `StubExtractor`, called from `extraction/pipeline.py`, `extraction/cli_evaluate.py`,
and exercised directly by ~25 test call sites across four test files. Adding a second parameter to
that signature (see below) touches every one of those. This is not a change contained to one file.

## Scope

In scope:
- `Extractor.extract()` gains a second parameter: `sections: dict[str, str]` — the paper's
  `paper_fulltext.sections` when a row exists, or `{}` when it doesn't. Every implementation
  (`HeuristicExtractor`, `SemanticExtractor`, `HybridExtractor`) and every call site is updated.
- `ClaimCandidate` gains a `section: str` field (default `"abstract"`), so the pipeline knows which
  section a candidate came from for grounding and persistence.
- A per-field → section-name mapping (below), used by both `HeuristicExtractor` and
  `SemanticExtractor` to search only the sections a field is likely to actually appear in.
- `extraction/pipeline.py`'s `_quote_is_grounded` checks against the right text (the named section,
  or `paper.abstract` for `section == "abstract"`) instead of always checking the abstract.
  `Evidence.section`/`source_locator` are persisted from the candidate, not hardcoded `"Abstract"`.
- `HeuristicExtractor`'s confidence for a full-text, named-section cue-phrase match becomes `"high"`
  instead of `"medium"` — per §15's own rule ("high: explicit statement in a named section"), not a
  new invented rule.
- `extraction/cli_evaluate.py` looks up each evaluated paper's `paper_fulltext` row (if any) and
  passes its `sections` through, so a benchmark paper with full text is evaluated the same way a
  real pipeline run would treat it.

Out of scope (deliberately deferred):
- Re-extracting the ~47,000 papers that already have `extracted_claims` from abstract-only runs.
  This slice only affects new extraction runs (new papers, or an explicit `--force` re-extract) going
  forward. Re-processing the existing corpus is a separate, expensive, and destructive decision
  (today's `--force` path deletes all evidence/claims corpus-wide) that deserves its own explicit
  ask later, not a side effect of this build.
- `StubExtractor` — synthetic test data, no real extraction logic to extend.
- Re-running the Sec 28 benchmark evaluation and comparing before/after numbers. This spec makes
  evaluation full-text-*capable*; actually re-running it and interpreting a score change is separate,
  later work once enough benchmark papers have full text fetched.
- Any change to `fulltext/` itself (already shipped) or to `analysis_claims`/`assessment/*` (already
  built on top of the abstract-only extraction this spec extends).

## Field → section mapping

Reused across `HeuristicExtractor` and `SemanticExtractor`, keyed on the same section names
`fulltext/parse.py` already produces (`introduction`, `related_work`, `methods`, `experiments`,
`results`, `discussion`, `limitations`, `conclusion`, `acknowledgments`, `references`, `body`):

```python
FIELD_SECTIONS: dict[str, tuple[str, ...]] = {
    "problem":            ("introduction",),
    "research_question":  ("introduction",),
    "method":             ("methods", "experiments"),
    "dataset":             ("methods", "experiments"),
    "main_contribution":  ("introduction", "conclusion"),
    "results":            ("results",),
    "limitations":        ("discussion", "limitations", "conclusion"),
    "research_gap":       ("conclusion", "discussion", "limitations"),
    "applications":       ("discussion", "conclusion", "introduction"),
}
```

For a field, the searchable text is every sentence from every section in its tuple that's actually
present in `sections` (a paper's real section set varies - `parse.py` falls back to a single `"body"`
bucket when no headings were recognized at all, and `FIELD_SECTIONS` intentionally has no `"body"`
entry - a field with no recognized section for it, or a paper with only a `"body"` bucket, falls back
to abstract-only search, today's exact behavior). `"references"`/`"acknowledgments"` are never
searched by any field - neither carries claim-bearing prose.

## Data flow (per paper, per field, per extractor)

```
extraction/pipeline.py::_process_paper(paper)
  -> look up paper_fulltext row for paper.id (one query, may be None)
  -> sections = row.sections if row else {}
  -> candidates = extractor.extract(paper, sections)

Inside HeuristicExtractor.extract(paper, sections):
  for each field in FIELD_SECTIONS:
      full_text_sentences = concat(split_sentences(sections[s]) for s in FIELD_SECTIONS[field] if s in sections)
      if full_text_sentences: search there first (per-field cue phrases), tag candidate section=<the
          section name the winning sentence came from>, confidence="high" on a cue-phrase hit
      else: fall back to searching paper.abstract exactly as today, section="abstract", confidence="medium"
  "problem" fallback (first sentence, confidence="low") stays abstract-only - full text's
      introduction is targeted above already; the abstract-opening-sentence heuristic doesn't have
      a full-text equivalent worth inventing.

Inside SemanticExtractor.extract(paper, sections):
  Preserves the original "each sentence proposes to its own single best-matching field" bipartite
  algorithm (this file's own docstring explains why - one generic-sounding sentence must not win
  several fields at once) by grouping fields BY THE EXACT POOL they'll draw from before running that
  algorithm, rather than resolving each field's full-text match in isolation:
    for each field in FIELD_SECTIONS:
        full_text_pairs = sentences_for_field(field, sections)
        if full_text_pairs: this field's pool_key = "fulltext:<the resolved section name>"
        else: this field's pool_key = "abstract" (shared by every field with no full-text match)
    group fields by identical pool_key
    for each group: run the ORIGINAL algorithm - one embed_texts() call over that group's shared
        sentence pool + only that group's field anchor queries, each sentence proposes to its best
        field within the group, each field picks its best proposing sentence, same MIN_SIMILARITY/
        MEDIUM_CONFIDENCE_SIMILARITY thresholds as today (confidence is NOT bumped to "high" for a
        semantic full-text hit - a similarity score isn't the same certainty as a verbatim cue-phrase
        hit, so only HeuristicExtractor gets that treatment); candidates from a "fulltext:*" group
        carry that section name, candidates from the "abstract" group carry section="abstract"
  A paper with no full text at all (sections={}) puts every field in the single "abstract" group,
  reproducing today's exact single-embed-call behavior - this is not a special case, it falls out of
  the grouping naturally. Two fields that happen to target the SAME section (e.g. method and dataset
  both -> methods/experiments) land in the same group and correctly compete for that section's
  sentences, preserving the collision-avoidance the original algorithm exists for. Cost stays
  bounded: each group only ever embeds its own resolved section's sentences (typically a few hundred
  words), never the whole paper.

Inside HybridExtractor.extract(paper, sections):
  unchanged _choose() logic, just threading `sections` through to both inner extractors' calls -
  the routing rules (literal cue-phrase over similarity, "problem" always prefers heuristic) don't
  need to know or care whether a candidate came from the abstract or a named section.

extraction/pipeline.py continues:
  for each candidate:
      haystack = paper.abstract if candidate.section == "abstract" else sections.get(candidate.section)
      _quote_is_grounded(candidate.evidence_quote, haystack) - same substring check as today, just
          against the right text
      on success: persist Evidence(section=candidate.section, source_locator=candidate.section, ...)
          instead of the hardcoded "Abstract"/"abstract" of today
```

## Backend changes

**`extraction/base.py`**:
```python
FIELD_SECTIONS: dict[str, tuple[str, ...]] = { ... as above ... }

@dataclass
class ClaimCandidate:
    claim_type: str
    claim_text: str
    evidence_quote: str
    confidence: str
    section: str = "abstract"

class Extractor(Protocol):
    extraction_method: str
    model_version: str
    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]: ...
```

**New shared helper**, `extraction/sections.py` (kept separate from `base.py` so `base.py` stays a
pure interface/types module, matching this codebase's existing file-per-responsibility convention):
```python
def sentences_for_field(field: str, sections: dict[str, str]) -> list[tuple[str, str]]:
    """Returns [(sentence, section_name), ...] for every sentence in the
    FIRST of field's FIELD_SECTIONS entries that's actually present in
    `sections` - section-list order is a priority, not a pool every
    section competes in equally (see "Section priority" below). Empty
    list if none of that field's target sections exist - the caller's
    cue to fall back to the abstract."""
```
Both `HeuristicExtractor` and `SemanticExtractor` call this once per field instead of duplicating
the "which sections, in what order" lookup logic.

**Section priority**: `FIELD_SECTIONS`' per-field tuple order is a priority, not a pool. For
`limitations` → `("discussion", "limitations", "conclusion")`, `sentences_for_field` returns only
`discussion`'s sentences if `discussion` is present in `sections`, falling through to `limitations`
then `conclusion` only when the earlier ones are absent - never merging sentences from multiple
sections for one field. This matches the mapping's own ordering intent (each tuple is already
written most-likely-to-least-likely) and keeps both extractors' "pick the best candidate" logic
operating over one section's sentences at a time, not a merged pool where a stray `conclusion`
sentence could outscore a better `discussion` one just because it happened to embed more similarly.

**`extraction/heuristic.py`**: `_first_matching_sentence` now also returns which section it matched
in; `extract()` tries `sentences_for_field(field, sections)` first, falls back to
`split_sentences(paper.abstract)` (today's path) only when that's empty. Confidence: `"high"` for a
cue-phrase hit in `sentences_for_field`'s output, `"medium"` for the existing abstract-only path
(unchanged from today), `"low"` for the `"problem"` first-sentence fallback (unchanged, abstract-only).

**`extraction/semantic.py`**: restructured around the pool-grouping algorithm above rather than a
per-field loop - `extract()` computes each field's `pool_key` (`sentences_for_field`'s resolved
section, or `"abstract"`), groups fields by it, and calls a `_match_pool(fields, sentences,
section_name)` helper (the original bipartite matching logic, extracted into its own method so it
can run once per group instead of once globally) for each group. The `MIN_SIMILARITY`/
`MEDIUM_CONFIDENCE_SIMILARITY` thresholds are reused unchanged - re-tuning them for full-text
sentence distributions is future work once real full-text extraction data exists to measure
against, not a blocking part of this build.

**`extraction/hybrid.py`**: `extract(self, paper, sections)` threads `sections` through to both inner
`.extract(paper, sections)` calls. `_choose()` needs one real fix: its non-"problem" branch currently
reads `if heuristic is not None and heuristic.confidence == "medium": return heuristic` - written
when "a literal cue-phrase hit" and `confidence == "medium"` were the same thing. They no longer are:
a full-text cue-phrase hit is now `confidence == "high"`. The check becomes
`heuristic.confidence in ("medium", "high")` - both values mean the same thing this rule cares about
("an explicit cue-phrase match exists"), so a `"high"`-confidence full-text heuristic candidate is
preferred over semantic similarity exactly like a `"medium"`-confidence abstract one always was,
instead of silently falling through to `semantic or heuristic` and losing that guarantee.

**`extraction/pipeline.py`**:
- `_process_paper` looks up `session.execute(select(PaperFullText).where(PaperFullText.paper_id == paper.id)).scalar_one_or_none()`
  once per paper, passes `.sections if found else {}` to `extractor.extract(paper, sections)`.
- `_quote_is_grounded(quote, haystack)` unchanged in implementation, just called with the right
  haystack (`paper.abstract` or `sections[candidate.section]`) based on `candidate.section`.
- `Evidence(section=candidate.section, source_locator=candidate.section, ...)` replaces the hardcoded
  `"Abstract"`/`"abstract"` literals.

**`extraction/cli_evaluate.py`**: the one direct external call site
(`extractor.extract(papers_by_source_id[a.source_id])`) becomes a small helper that looks up that
paper's `paper_fulltext` row the same way the pipeline does, and passes `sections` through.

## Testing

Following this codebase's established per-module test split:
- `extraction/sections.py`: pure-function tests for `sentences_for_field` - present sections, absent
  sections, a field with no `FIELD_SECTIONS` target sections present (empty result), body-only
  fallback (empty result, since `"body"` is never a mapped target).
- `heuristic.py`/`semantic.py`/`hybrid.py`: extend each existing test file's ~6-10 `extract()` calls
  to pass `sections={}` (preserves every existing abstract-only assertion unchanged) plus new cases
  with a real `sections` dict proving a full-text match wins over an abstract match for the same
  field, and that confidence is `"high"` (heuristic) for a full-text cue-phrase hit.
- `pipeline.py`: extend `FakeExtractor` (test double) to accept `(paper, sections)`; add cases
  confirming a `PaperFullText` row is looked up and passed through, grounding checks the right
  section's text (not always the abstract), and `Evidence.section` persists the real section name.
- `cli_evaluate.py`: a test confirming a benchmark paper with a `paper_fulltext` row gets its
  sections passed to `extractor.extract`, and one without gets `{}` - same fallback contract as the
  live pipeline.

## Migration safety

No schema change - `ClaimCandidate.section` and the `Extractor.extract()` signature are Python-level
changes only. Fully backward compatible in behavior: any paper without a `paper_fulltext` row (the
vast majority of the corpus until `rb-fulltext-fetch` processes more of it) extracts identically to
today, field-by-field, confidence-by-confidence. `reset_extraction_data`/the `--force` re-extract path
is unchanged; running it after this ships would produce the full-text-aware version's output for
papers that happen to have full text yet, and today's abstract-only output for the rest.

## Open questions for whoever implements this

1. `MIN_SIMILARITY`/`MEDIUM_CONFIDENCE_SIMILARITY` in `semantic.py` were calibrated against the real
   40-paper benchmark's abstract-only similarity distribution (§ that module's own docstring). Once
   full-text extraction has run against enough real papers, it's worth checking whether full-text
   sentence similarity scores cluster differently than abstract sentence scores did - if they do,
   these thresholds may need separate calibration for the full-text path rather than reusing the
   abstract-tuned numbers. Not blocking for this build (reusing them unchanged is the explicit
   in-scope decision above), just flagged for whoever eventually re-runs Sec 28 evaluation.
