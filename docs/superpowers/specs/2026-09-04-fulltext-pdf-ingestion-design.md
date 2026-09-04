# Full-Text PDF Ingestion Pipeline — Design Spec

Source of truth for the gap this fills: blueprint §46 ("Future PDF Processing") and §3's decision
to start Phase 1 with structured-API metadata/abstracts only, deliberately deferring full-text PDF
processing. That deferral was correct for Phase 1; this spec is the "when justified" follow-up —
the corpus now has real extraction, retrieval, gap detection, and assessment pipelines running
against abstract-only text, and abstracts cap how much any of them can ground claims in.

## Why this isn't started with GROBID

§46 gives two options: PyMuPDF (already a dependency, already proven on the assessment-upload path
— see `api/assessment_routes.py`'s `.pdf` handling) or GROBID (a separate Java/Docker service with
real structural PDF parsing). §46 explicitly frames GROBID as "only when justified," and §38
("Minimize Infrastructure") argues against adding a new service without a measured need. This spec
uses PyMuPDF only. Section boundaries come from heuristic heading matching, not structural parsing
— less accurate than GROBID's TEI-XML output, but zero new infrastructure, zero new disk/RAM
budget, and reuses code already battle-tested on real uploaded PDFs.

## Scope

In scope:
- A new `fulltext` package: resolve each paper's PDF URL, fetch it, parse it into heuristic
  sections via PyMuPDF, and persist the result.
- Source coverage: arXiv, CORE, and Semantic Scholar (all three have a genuinely fetchable PDF URL
  when `open_access` is true — see "Per-source PDF URL resolution" below), plus a best-effort
  attempt on Springer even though its `url` field is an HTML landing page, not a PDF — parsing
  failures are caught and logged rather than silently skipped or crashing the run.
- Run-history tracking (`fulltext_fetch_runs`/`fulltext_fetch_errors`), idempotent by default,
  `--force` to re-fetch, matching every other pipeline's shape in this codebase
  (`ExtractionRun`/`EmbeddingRun`/`CitationFetchRun`/`GapDetectionRun`).
- A CLI (`rb-fulltext-fetch`) and one admin-panel trigger route, matching the existing 8 pipeline
  triggers in `admin_routes.py`.

Out of scope (deliberately deferred — see "Two-slice split" below):
- Any change to `extraction/pipeline.py`'s grounding check or to any `Extractor` implementation.
  Every current extractor (`HeuristicExtractor`, `HybridExtractor`, `SemanticExtractor`) is built
  entirely around `paper.abstract` — sentence splitting, cue-phrase matching, semantic similarity —
  not just the grounding check. Extending them to also draw candidates from full-text sections is a
  substantial rewrite of three extractor implementations and their test/benchmark suites, not a
  small tweak. That's real, separate follow-up work once full text actually exists in the database
  to test against.
- GROBID, OCR, or any parser beyond PyMuPDF.
- Springer-specific PDF-URL resolution (its API has a real content-download endpoint, distinct from
  the metadata endpoint already wired in `connectors/springer.py`) — a distinct, separate piece of
  work if Springer's best-effort success rate turns out too low to be useful.

## Two-slice split (matching this session's `analysis_claims` precedent)

Slice 1 (this spec): fetch + parse + store full text as a new, independently useful capability with
no consumer yet — same "ship the layer, wire consumption later" pattern already used for
`analysis_claims` (Sec 16) earlier in this project. Slice 2 (future, separate design): extend the
extractors to actually use it.

## Per-source PDF URL resolution

| Source | `paper.url` today | PDF URL for this pipeline |
|---|---|---|
| arXiv | abstract landing page (`arxiv.org/abs/<id>`) | derived from `paper.source_id`: `https://arxiv.org/pdf/<source_id>` — a stable, documented URL pattern, not a scrape |
| CORE | already `downloadUrl`/fulltext link (`connectors/core.py:160-163`) | `paper.url` as-is |
| Semantic Scholar | the `openAccessPdf.url` when `open_access` is true (`connectors/semantic_scholar.py:171-193`) | `paper.url` as-is |
| Springer | HTML landing page (`connectors/springer.py:181`, `html_url`) | `paper.url` as-is, attempted anyway — expected to fail parsing for most/all records until a dedicated resolver exists |

`resolve_pdf_url(paper) -> str | None` returns `None` for any paper where `paper.open_access` is
not `True` — this pipeline only ever attempts open-access papers, consistent with §4's licensing
constraints.

## Data flow

```
rb-fulltext-fetch --all --save  (or the admin panel's "fetch full text" trigger)
  -> select papers where open_access = true and no paper_fulltext row exists yet (unless --force)
  -> for each paper:
       url = resolve_pdf_url(paper)
       if url is None: skip, count as papers_skipped_no_url
       else:
         try: response = requests.get(url, timeout=30)
         except (network/HTTP error): log to fulltext_fetch_errors, count as papers_failed, continue
         try: sections = parse_pdf(response.content)
         except (pymupdf.FileDataError / empty result): log to fulltext_fetch_errors, count as
              papers_failed, continue
         persist PaperFullText(paper_id, sections, source_url=url, fetch_method="pymupdf")
         count as papers_fetched
  -> record run totals on fulltext_fetch_runs, same incremental-update shape as
     GapDetectionRun/CitationFetchRun (_sync_run pattern)
```

## Backend

**New package** `src/researchbridge/fulltext/`:

```python
# pdf_url.py
def resolve_pdf_url(paper: Paper) -> str | None:
    """None if paper.open_access is not True. Otherwise per-source resolution
    per the table above; arXiv derives from source_id, everything else uses
    paper.url as-is (Springer's is expected to often fail downstream)."""

# parse.py
def parse_pdf(pdf_bytes: bytes) -> dict[str, str]:
    """Raises PdfParseError (wrapping pymupdf.FileDataError, or an
    empty-text result) on anything that isn't a real parseable PDF with
    extractable text. On success: splits full text into sections by
    matching common heading lines (Introduction, Method(s)/Methodology,
    Results, Discussion, Limitations, Conclusion, References, ...) against
    a fixed heading vocabulary, case-insensitive, anchored to line starts.
    Falls back to {"body": <full text>} if no heading matches at all -
    still useful, just unsectioned."""

class PdfParseError(Exception): ...

# pipeline.py
class FullTextFetchPipeline:
    """Structurally identical to ExtractionPipeline (extraction/pipeline.py):
    idempotent paper selection, per-paper try/except that logs to
    fulltext_fetch_errors and continues rather than crashing the run,
    incremental run-history updates via the same _sync_run shape as
    gaps/batch.py and citations/batch.py."""
    def __init__(self, session_factory: sessionmaker[Session]) -> None: ...
    def run(self, limit: int | None = None, force: bool = False) -> str: ...

# cli.py
def main() -> None:
    """argparse: --limit, --force. No --extractor equivalent - there's
    only one parser (PyMuPDF) in this slice."""
```

## Schema (one migration)

```sql
paper_fulltext
--------------
id              UUID PRIMARY KEY
paper_id        UUID NOT NULL REFERENCES papers(id), UNIQUE
sections        JSONB NOT NULL   -- {"introduction": "...", "results": "...", "body": "...", ...}
source_url      TEXT NOT NULL    -- the URL actually fetched, for provenance/debugging
fetch_method    VARCHAR NOT NULL DEFAULT 'pymupdf'
fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()

fulltext_fetch_runs   -- same shape as gap_detection_runs / citation_fetch_runs
----------------------
id, status, started_at, finished_at,
papers_seen, papers_fetched, papers_skipped_no_url, papers_failed,
error_summary, force

fulltext_fetch_errors   -- mirrors extraction_errors
---------------------
id, fulltext_fetch_run_id FK, paper_id FK, error_type, error_detail, occurred_at
```

`paper_fulltext.paper_id` UNIQUE enforces one row per paper (re-fetch under `--force` updates the
existing row rather than inserting a duplicate — same upsert-or-delete-then-reinsert choice
`ExtractionPipeline`'s `reset_extraction_data` makes, scoped down to this table only).

## CLI + admin wiring

`pyproject.toml`: `rb-fulltext-fetch = "researchbridge.fulltext.cli:main"`, alongside the existing
`rb-extract`/`rb-embed`/`rb-gaps-detect` entries.

`api/admin_routes.py`: one more entry in `PIPELINE_KEYS`, one more `POST
/api/admin/pipelines/fulltext/trigger` route firing `rb-fulltext-fetch --all --save` via the
existing `pipeline_triggers.trigger()` subprocess mechanism — identical shape to
`trigger_citations_fetch`/`trigger_embedding`.

## Error handling

- Non-open-access paper: never attempted (`resolve_pdf_url` returns `None` before any network
  call) — counted as `papers_skipped_no_url`, not an error.
- Network/HTTP failure (timeout, 404, connection error): logged to `fulltext_fetch_errors` with
  `error_type="fetch_error"`, run continues to the next paper.
- Non-PDF or unparseable response (Springer's landing pages, a broken link, a paywall redirect that
  returns an HTML "subscribe" page): `parse_pdf` raises `PdfParseError`, logged with
  `error_type="parse_error"`, run continues.
- Already-fetched paper without `--force`: skipped silently (same idempotency contract as
  `ExtractionPipeline`'s `_select_unprocessed_papers`).
- One paper's failure never aborts the run — same "fail loud per-item, keep going" contract as
  every other pipeline in this codebase (§8's ingestion-reliability requirement, extended here).

## Testing

Following `extraction/pipeline.py`/`test_extraction_pipeline.py`'s established split:
- `pdf_url.py`: table-driven test of `resolve_pdf_url` per source (arXiv derivation, CORE/Semantic
  Scholar pass-through, Springer pass-through, `None` when not open-access) — pure function, no I/O.
- `parse.py`: real small PDF fixtures (a synthetic PDF with clear headings, one with no headings at
  all, one that's actually plain text/HTML mislabeled as PDF triggering `PdfParseError`) — mirrors
  how `assessment_routes.py`'s upload tests already exercise `pymupdf.FileDataError`.
- `pipeline.py`: `requests.post`/`get` mocked (same pattern as `test_citations_fetch_batch.py`) —
  happy path persists a `PaperFullText` row, fetch failure logs an error and continues, parse
  failure logs a different error and continues, `--force` re-fetches an already-fetched paper,
  idempotency skips one that's already fetched without `--force`, non-open-access papers are never
  attempted.
- CLI/admin route tests: thin, mirroring `test_gaps_cli_detect.py`/the existing trigger-route tests
  in `test_admin_api.py`.

## Migration safety

Fully additive: one new migration creating three new tables, zero changes to any existing table,
zero changes to `extraction/pipeline.py` or any `Extractor`. Running (or never running)
`rb-fulltext-fetch` has no effect on any existing pipeline's behavior — `paper_fulltext` has no
consumer in this slice, exactly like `analysis_claims` had no reader until this session's follow-up
work added one.

## Open questions for whoever implements this

1. Heading vocabulary for section-splitting: this spec doesn't enumerate the exact regex/heading
   list. Worth checking a handful of real arXiv PDFs (this codebase already ingests plenty) to see
   what heading text actually appears before finalizing the list — LaTeX-generated PDFs are fairly
   consistent, but common variants ("Related Work" vs "Background", "Discussion" vs "Discussion and
   Future Work") should be enumerated from real examples, not guessed.
2. Springer's real-world success rate through this best-effort path is unknown until it's tried
   against the actual corpus — if it turns out to be near-zero, that's useful evidence for whether
   the "distinct Springer PDF resolver" follow-up (mentioned in Out of scope) is worth building.
3. Whether corpus-wide `rb-fulltext-fetch --all` should run automatically as part of the existing
   ingestion pipeline (fetch full text right after a paper is ingested) or stay a fully separate,
   manually-triggered pass indefinitely. This spec assumes separate/manual, matching how
   `rb-extract`/`rb-embed`/`rb-gaps-detect` are all separate manual stages today rather than chained
   automatically.
