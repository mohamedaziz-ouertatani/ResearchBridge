# Assessment Report Hardening — Design Spec

Source of the bug list: an audit of 10 exported `assessment-*.md` files (in
`~/Downloads`, not part of this repo) surfaced nine symptoms spanning
extraction quality, rendering duplication, header/body consistency, and
generation gaps. This spec maps each symptom to its confirmed root cause in
`src/researchbridge/` and defines the fix. Investigation included live
read-only queries against the dev Postgres instance for the two samples
showing the header/body mismatch.

## Investigation summary (what's real vs. what's stale data vs. what's by design)

Three of the nine symptoms turned out **not** to be live code bugs once traced
to ground truth:

- **Header count 0 / missing References (symptom 3, 9):** confirmed via live
  DB query that the two affected assessments (`7e2c3b87…`, `b2470ff7…`) were
  built **2026-09-03/04, before** the existing-solutions relevance-gate fix
  (`0f43541`, Sept 4 21:26) and the application-relevance filter that
  followed it. Their `research_assessment_evidence` junction table has zero
  rows across every role even though the underlying `evidence` rows for
  their retrieved papers are real (`extraction_method='hybrid'`, not
  `'stub'`). Every currently-built assessment (the 10 most recent, all Sept
  7) has 19 linked evidence rows. This is stale historical data from a
  since-hardened pipeline, not a reachable bug in current code — but nothing
  today would prevent a *future* regression from reintroducing it, which is
  why Phase 3 adds a build-time invariant, not just a backfill.
- **"No relevant paper" in Applications/Risks despite Existing-solutions
  citing papers (symptom 4):** the sample that shows this (`b2470ff7…`) is
  the *same* stale pre-hardening assessment above. Current code applies one
  shared `RELEVANCE_DISTANCE` gate consistently across
  `existing_solutions.py`, `applications.py`, and `risks.py` against the
  same `papers_by_distance` list built once per `build_assessment()` call —
  this specific mismatch isn't reachable from current code. Phase 3's
  backfill resolves the visible symptom; Phase 4 hardens the *display* of
  *why* a field is null regardless.
- **"High feasibility from 1 paper" (part of symptom 5):** `feasibility.py`'s
  `distinct_count` is a paper-id-keyed count, already deduplicated by
  construction — the documented rule is 1 paper → `medium`, 2+ → `high`, and
  every sample file's actual data matches that rule (e.g. `bf1be977…` shows
  "high" with 4 distinct papers, correctly). Phase 7 is a verification pass,
  not a fix, unless testing turns up a real counterexample.

The other six map to confirmed, fixable root causes in current code. Two
scope corrections were made after the initial fix proposal, per discussion:

- **Opportunities synthesis already runs inline** during `build_assessment()`
  when `enable_llm_stages=True` (`build.py:148-194`), and `ollama_enabled()`
  already defaults to `True` (changed 2026-09-05, see
  `opportunity_synthesis.py`'s own docstring). All 10 samples — including
  fresh Sept 7 ones — still show the static refusal, so either Ollama wasn't
  reachable when these were generated, or `applications.status` genuinely
  wasn't `"found"` for these idea-only inputs (plausible: idea-only inputs
  rarely have abstracts stating an application outright). Phase 4 is
  therefore about **diagnosing and displaying which of those is true**, not
  flipping a flag that's already flipped.
- **Applications/Risks stay extraction-only.** Both are deliberately
  grounded in a paper's own stated claim, never synthesized — the same "no
  Grounding Illusion" principle `opportunities.py`'s docstring cites as the
  reason opportunities synthesis needed a whole separate,
  citation-validated, fail-closed subsystem. Extending invented content into
  Applications/Risks would reintroduce exactly the hallucination risk that
  subsystem exists to fence off. Phase 4 makes the *reason* a field is null
  diagnostic; it does not add invention where there was none.

## Phase 1 — Quote quality gate (symptom 1)

**Root cause:** no filter anywhere in the extraction path rejects a quote for
being too short, ending mid-word/mid-sentence, or being a copyright/
author-email/bare-heading line.

- `extraction/sentences.py:14,17-18` splits on `re.compile(r"(?<=[.!?])\s+")`
  with no length or completeness check.
- `extraction/heuristic.py` (`_CUE_PHRASES`-based first-match) and
  `extraction/semantic.py` (`MIN_SIMILARITY = 0.15` nearest-to-anchor match)
  both pick a "sentence" with no shape validation — `semantic.py`'s own
  docstring already documents that embedding similarity can prefer a
  "fluent-but-content-free" sentence over the real content one.
- `extraction/validation.py` checks claim-*type* language (does this read
  like a `research_gap`/`method`/etc.), not quote *shape* — and
  `research_question`/`main_contribution` skip it entirely (accepted
  unconditionally, per that module's own docstring).
- `extraction/pipeline.py:231-238` (`_quote_is_grounded`) only checks
  exact-substring grounding — a truncated fragment like `"We propose DU-"`
  is a valid substring of its source text and passes trivially.
- `fulltext/parse.py:65-95` (`split_sections`) never strips copyright lines,
  author/affiliation blocks, or repeated running headers from PDF text —
  they land in whatever section they physically fall in and are eligible to
  be selected like any other sentence.

**Fix:** new `extraction/quote_quality.py` module, called from
`heuristic.py`, `semantic.py`, and `hybrid.py` immediately after a candidate
sentence is chosen and before it's returned as a claim:

- Reject if under a minimum token count (6, chosen to admit genuine short
  technical sentences while rejecting single-word/two-word fragments like
  `"For"` — tunable during implementation against the sample corpus).
- Reject if it doesn't end in terminal punctuation (`.`, `!`, `?`, or a
  closing quote/paren following one) — catches mid-sentence PDF line-wrap
  cutoffs.
- Reject if it matches a copyright/license pattern (`©`, `All rights
  reserved`, `Publisher Copyright`), an email/correspondence-block pattern
  (`Correspondence to:`, a bare email address as most of the line), or looks
  like a bare section heading (Title Case with no verb, or a numbered
  section pattern like `8.1.` alone).
- Reject if it ends in a hyphenated word fragment (PDF line-wrap
  mid-token break, e.g. `"We propose DU-"`).

A rejected candidate falls through to the next-best candidate for that
field (heuristic/semantic already have fallback ordering — this just adds
one more disqualifying check before acceptance), or to `not_assessed` if
nothing passes, consistent with the existing "NULL over fabricated
certainty" principle (`build.py`'s own module docstring, Sec 22).

## Phase 2 — Dedup before rendering (symptom 2)

**Root cause, confirmed structurally in `assessment/export.py`'s
`build_markdown()` (lines 997-1026):**

For every section, if `section.body` is set, it's rendered once (as short
blockquotes via `_md_comparison_summary()` for "Existing solutions", or as
one paragraph otherwise) — and then, unconditionally, `section.evidence` is
looped and rendered *again* as long blockquotes with a markdown link. For
"Existing solutions" specifically this is a **guaranteed** duplicate, not a
coincidental one: `existing_solutions.py:82-93` builds `comparison_summary`'s
text and its `evidence_ids` list from the exact same claim-iteration loop,
so every quote embedded in the short-blockquote body is, by construction,
also present in `section.evidence`.

Separately, within `section.evidence` itself, nothing dedupes identical
`evidence_id`s attached under the same role more than once — this is the
mechanism behind the "same quote 3x in a row" pattern seen in Novelty
assessment (dimension-coverage matching in `coverage.py` can cite the same
top-matching claim for more than one dimension, and `build.py`'s evidence
linking has no distinctness check when flattening these into one role's
list).

**Fix:**
- In `build_markdown()` (and the equivalent docx/pdf paths), when a section
  renders its body via `_md_comparison_summary()`, skip the redundant
  `section.evidence` loop for that section — the body rendering already
  covers every evidence_id in that list, in the same order.
- Add a dedup-by-`evidence_id` step (order-preserving, first-seen wins) when
  `by_role` lists are built in `build_report_sections()`, so no role's
  evidence list can contain the same evidence_id twice regardless of how
  many claim/dimension-coverage paths cited it.

## Phase 3 — Evidence-linking invariant + backfill (symptoms 3, 9)

- Add an assertion/log in `build.py`, right after each `assess_*` call and
  before `session.add(assessment)`: if a field's narrative text is non-null,
  its evidence-ID list must be non-empty. Raise loudly (test/dev) or log at
  ERROR with the assessment's input id (prod) rather than silently
  persisting a row that can't stand behind its own text — this is the
  concrete guard that keeps the Sept 3/4 failure mode from recurring
  unnoticed.
- One-off backfill script (`scripts/`) that finds assessments older than the
  Sept 4 21:26 hardening commit with an empty
  `research_assessment_evidence` set and non-null narrative text, and
  re-runs them through the existing rerun path
  (`POST /assessments/{id}/rerun`, `assessment_routes.py:361`) so they pick
  up current-code evidence linking. Confirmed candidates from this
  investigation: `7e2c3b87-ea02-4942-9671-3770329eacdb`,
  `b2470ff7-4888-494e-8c1e-3780ec15d038`; the script should scan for any
  others matching the same pattern rather than hardcoding these two IDs.
- Defensive fallback in `_related_papers()`/References rendering: if
  `assessment.evidence` is empty but `assessment.retrieved_paper_ids` is
  non-empty, this now indicates a data-integrity problem the Phase 3
  invariant should have caught at write time — Phase 8 turns this into an
  explicit validation failure rather than a silently-empty References
  section.

## Phase 4 — Diagnostic null-reasons (symptoms 4, 7)

- `opportunity_synthesis.py`/`build.py`: distinguish, in what's persisted
  and what's shown, between (a) Ollama disabled, (b) Ollama unreachable/
  timed out after retry, (c) response failed validation after retry, (d)
  `applications.status != "found"` so synthesis was never attempted. Today
  all four collapse into one static `OPPORTUNITIES_REASON` string
  (`export.py:43-46`). Add a `potential_opportunities_status` (or reuse/
  extend an existing status field) that export.py keys off the same way
  `_APPLICATIONS_UNASSESSED_REASONS` already does, so a human reviewer can
  tell "retry this later, Ollama was down" from "there's genuinely nothing
  to build an opportunity from" from "no reviewer has requested this yet."
- Fix `_APPLICATIONS_UNASSESSED_REASONS.get(status, default)`
  (`export.py:127-129`): this silently falls back to the "no relevant paper"
  message for *any* unrecognized status string. Change to fail loud (raise
  or log ERROR) on an unrecognized status instead of masking a status-value
  drift as the wrong user-facing message.
- Verify/document the Ollama deployment for whatever environment produces
  exported reports (confirm `OLLAMA_ENABLED`/`OLLAMA_HOST` reach a live
  model) — an operational check, not a code change, but necessary since the
  code path for opportunities synthesis already exists and defaults on.

## Phase 5 — Semantic dimension extraction via LLM (symptom 8)

**Root cause, confirmed in `assessment/dimensions.py`:** `extract_dimensions()`
is a from-scratch RAKE implementation — candidate phrases are runs of
non-stopword tokens (`_candidate_phrases()`), scored by
`degree(word)/frequency(word)` word co-occurrence
(`_score_words()`), with no semantic understanding of the idea. This is a
deliberate, documented architectural choice (see the module's own docstring
and `docs/superpowers/plans/2026-08-29-dimension-aware-assessment-coverage.md`)
made specifically to avoid an LLM dependency for this stage — so this phase
reverses a considered tradeoff, not an oversight, and must preserve the
"never fabricate a dimension not implied by the input" invariant when doing
so.

**Fix:** add an LLM-based extraction path (same Ollama integration pattern
as `opportunity_synthesis.py`: local call, one retry, fail closed) that asks
the model for 5-8 real technical concepts from the idea text, each grounded
in the input's own wording (not invented terminology). On
`ollama_enabled()`-false or synthesis failure after retry, fall back
automatically to the existing RAKE implementation — this keeps dimension
coverage available with zero external dependency exactly as it is today
whenever the LLM path isn't usable, matching every other fail-open/
fail-closed precedent in this codebase (`application_relevance.py` fails
open, `opportunity_synthesis.py` fails closed to NULL; dimensions has a
safe non-null fallback available, so it fails open to RAKE rather than to an
empty dimension list).

## Phase 6 — Gap-quality tiering (symptom 6)

**Root cause, confirmed in `gap.py:201-244`
(`_explicit_research_gap_claim`) and `extraction/validation.py:592-604`:**
a paper's `research_gap`-type claim carries a `validation_tier` of
`"strong"` (unambiguous gap language, e.g. "remains an open problem") or
`"weak"` (boilerplate future-work language matched by
`_WEAK_GAP_LANGUAGE_RE`, e.g. "future work will..."). `gap.py` already
threads `is_strongly_stated = (validation_tier == "strong")` into
`assess_recommendation()`'s confidence calculation (`build.py:197-207`) —
but the *displayed* `text` is the raw claim text regardless of tier, and
`_explicit_research_gap_claim()` returns the **first** relevant paper
(nearest-first) with *any* research_gap claim, strong or weak, without
checking whether a strong-tier claim exists among other relevant papers
first.

**Fix:**
- Change `_explicit_research_gap_claim()`'s selection order: prefer the
  nearest relevant paper with a **strong**-tier claim; only fall back to the
  nearest weak-tier claim if no strong-tier claim exists among the relevant
  set.
- When a weak-tier claim is the one selected (i.e., no strong claim was
  available), prefix the displayed text to make the tier explicit instead of
  presenting generic future-work language as a confident finding — e.g.
  `"No explicit gap stated; nearest related paper mentions future work
  generically: \"{text}\""` instead of the current unqualified `"Explicitly
  stated in \"{title}\": \"{text}\""`.
- No change to `_inferred_cross_paper_gap()` or the recommendation-confidence
  wiring — `is_strongly_stated` already flows there correctly; this phase
  only changes what's selected and how it's labeled for *display*.

## Phase 7 — Feasibility recount verification (symptom 5)

`feasibility.py:228` computes `distinct_count = len(close_titles) +
len(widened_titles)`, both paper-id-keyed collections built by iterating
`close_ids`/`widened_ids` once per distinct paper — already deduplicated by
construction, independent of Phase 2's evidence-row dedup (a duplicate
evidence row for the same paper can't inflate a paper-count). Every sample
file's feasibility level matches the documented 1-paper→medium,
2+-papers→high rule exactly.

This phase is a **verification pass during implementation**, not a planned
code change: re-run the sample inputs against post-Phase-1/2/3 code and
confirm `distinct_count` still matches the displayed level. If testing
surfaces a real counterexample (rather than the datasets used in the audit,
which don't contain one), fix `distinct_count`'s computation directly; only
add code if verification finds something.

## Phase 8 — Persistence-time completeness check (ties symptoms 3, 9 together long-term)

**Root cause:** no completeness or cross-field-consistency validation exists
anywhere in the write path. `build.py:215-253` constructs the
`ResearchAssessment` ORM row directly from each `assess_*` result and
immediately `session.add()`/`flush()`s it. `ResearchAssessmentOut`
(`api/schemas.py`) validates field *types* at the API-response boundary, not
cross-field invariants — it happily accepts `evidence=[]` next to a
non-null `comparison_summary` full of quotes. `export.py:1028`'s References
section is conditioned on `related` (itself derived from `assessment.evidence`),
so an assessment with real body text but empty evidence silently drops the
whole section with no error anywhere in the chain.

**Fix:** a single completeness-check function, called from `build.py`
immediately before persisting (subsumes Phase 3's narrower invariant into
one general check covering every field, not just the ones investigated
here):
- For every field with non-null narrative text, its corresponding evidence-
  ID list is non-empty.
- `len(assessment.evidence)` (header count) is derivable from the same
  evidence rows the body sections render — already true by construction in
  current code (`build_report_sections()` and `_stats_tiles()` both read
  `assessment.evidence` in the same function call), this check exists to
  keep that true as the codebase evolves.
- On failure, raise before the row is persisted (fail loud, not a silently
  broken row that surfaces as a confusing export three days later) — this
  is the structural fix that makes another Phase-3-shaped incident
  unreachable rather than merely caught after the fact.

## Testing

- Extraction quality: unit tests for `quote_quality.py` against the exact
  fragments from the audit (`"For"`, `"We propose DU-"`, the Springer
  copyright line, the Jiangchao Yao correspondence block, `"Sequential
  Consistency of Attention Heads"`) — each must now be rejected.
- Dedup: a `build_markdown()` snapshot test against an assessment fixture
  with a comparison_summary + matching evidence_ids (the exact
  `existing_solutions.py` shape) — no quote should appear twice in output.
- Evidence-linking invariant: a `build_assessment()` test that forces an
  `assess_*` function to return non-null text with empty evidence_ids and
  asserts the build now fails loudly instead of persisting.
- Gap tiering: unit tests over both a strong-tier-available case and a
  weak-tier-only case, asserting the displayed text and its prefix differ
  correctly.
- Dimension extraction: a fixture-based test comparing RAKE-only output
  (today's behavior, kept as the fallback) against the new LLM path when
  Ollama is mocked available/unavailable, confirming automatic fallback
  actually engages on failure.
- Backfill script: dry-run against a copy of the dev DB, confirming exactly
  the two known-stale assessments (plus any others matching the same
  pre-hardening-timestamp + empty-evidence pattern) are identified before
  actually calling rerun.

## Rollout order

Phases 1-3 (quality gate, dedup, evidence invariant + backfill) are
independent of each other and of Phases 4-8, and address the most visibly
broken symptoms (garbled quotes, literal duplication, missing References) —
implement and ship first. Phase 8 subsumes Phase 3's invariant into a
general check and should follow once Phase 3's narrower version is proven
in production. Phases 4-6 (diagnostic statuses, LLM dimension extraction,
gap tiering) are independent of each other and of 1-3/8, and can ship in any
order after. Phase 7 is verification-only and runs alongside whichever
phase's testing exercises `feasibility.py`'s inputs (naturally covered
once Phase 2's dedup is in place, since that's the only phase that touches
evidence counts feasibility depends on).
