# Math-Aware Benchmark Extraction (Nougat) — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint). This spec covers a narrow, self-contained addition to the existing benchmark/annotation workbench slice (`benchmark/fulltext.py`, `rb-benchmark-fetch`, `/api/benchmark`, `app/annotate/[sourceId]`) — never wired into the main ingestion pipeline, exactly like the PyMuPDF extraction it sits alongside.

## Why

`extract_text()` (PyMuPDF) reads PDF glyph positions, not semantic structure. Math-heavy papers come out visibly scrambled — verified live against a real cached benchmark paper (`1812.02641`, "Local Conditioning in Undirected Networks"): a 2×1 matrix expression and inline exponents come back as disconnected single-character lines, and `∑`/`∏` big-operator glyphs literally extract as the Latin letters `X`/`Y`. This was already a known, documented limitation (see the module's existing `_UNRENDERABLE` comment) — this spec adds a second, math-aware extractor for the same 40 papers so annotators reading proof-heavy or notation-heavy papers aren't working from scrambled text.

## Scope

In scope:
- A second extractor, Nougat (`nougat-ocr`), selectable alongside PyMuPDF behind the same `extract_text()` interface — **not a replacement**. PyMuPDF's implementation is untouched.
- Per-extractor cached output, stored under different filenames, so re-running with one extractor never overwrites the other's cached file.
- `rb-benchmark-fetch --extractor {pymupdf,nougat} --force` — `--extractor` selects which engine runs (default stays `pymupdf`, so existing behavior is unchanged unless explicitly overridden); `--force` bypasses the existing skip-if-cached check, needed because Nougat output must land in its own file, not silently reuse a stale PyMuPDF cache-hit check.
- The annotation workbench renders whichever extractor's output is available, preferring Nougat when both exist, with a toggle to switch and read the other — this is the "compare both methods" mechanism. Nougat's Markdown+LaTeX renders as real typeset math (`react-markdown` + `remark-math` + `rehype-katex`); PyMuPDF's plain text keeps rendering in the existing `<pre>` block.
- Text-selection → evidence capture re-verified against the rendered Markdown DOM (not just the old plain-text `<pre>`), since `window.getSelection()` now reads over real HTML elements (headers, `<strong>`, KaTeX's own DOM) instead of one flat text node.

Out of scope (deliberately deferred):
- Any change to the main ingestion pipeline's PDF handling (arXiv/Springer/Semantic Scholar) — this touches only the 40-paper benchmark slice.
- Mathpix or any other extractor — Nougat only, per the cost/data-locality tradeoff already decided.
- An automated comparison/scoring tool (e.g. diffing the two outputs, a quality metric). The "evaluation" is a human reading both renders side-by-side in the workbench — consistent with this benchmark's existing principle that human annotation is the ground truth, not a computed score.
- GPU acceleration / performance tuning. CPU-only, accepted as slow, one-time.

## Backend design

**`benchmark/fulltext.py`**:
- `fulltext_path(output_dir, source_id, extractor="pymupdf") -> Path` — extractor-aware filename. `pymupdf` keeps the existing `{source_id}.txt` (the 40 already-cached files stay valid, untouched, still found by a default call); `nougat` writes to `{source_id}.nougat.md` (the extension signals Markdown, distinguishing it from the plain-text convention).
- `extract_text(pdf_bytes, extractor="pymupdf") -> str` — dispatches to `_extract_pymupdf(pdf_bytes)` (today's implementation, moved verbatim, behavior-identical) or `_extract_nougat(pdf_bytes)` (new). Nougat's import (`torch`, the `nougat` package) is lazy, inside `_extract_nougat`, exactly like PyMuPDF's own lazy-import reasoning today ("the rest of the benchmark tooling doesn't pay for [it]") — now far more important, since `torch`+model checkpoint is a multi-GB, slow-to-import dependency the rest of the app must never pay for.
- `_extract_nougat`: rasterizes PDF pages (Nougat's own preprocessing pipeline handles this internally) and runs a checkpoint, returning the model's Markdown output. The exact checkpoint name/selection, model-loading call, and download/caching behavior are **not assumed here** — verified against whatever `nougat-ocr` version actually installs during implementation (its API has changed across releases), and documented in the module docstring the same way the Springer/Semantic Scholar connectors document their live-verified gotchas. Default to whatever the installed package treats as its own default checkpoint rather than hardcoding a name that may not exist in that version.
- `_tidy()` gets reviewed against real Nougat Markdown output during implementation — blank lines are structurally meaningful in Markdown (paragraph breaks) in a way they aren't in raw PDF text extraction, so the existing blank-line-collapsing regex may need a Markdown-aware adjustment. Verified live, not assumed.
- `fetch_fulltext(source_id, output_dir, session=None, extractor="pymupdf", force=False) -> str` — same shape, two new params. Writes to the extractor-specific path; `force=True` skips the exists-check.

**`ingestion/cli_fetch.py`** → `--extractor` (choices: `pymupdf`, `nougat`; default `pymupdf`) and `--force` (default `False`) CLI flags, threaded through to `fetch_fulltext`. The existing skip-if-cached log line stays accurate per-extractor.

**`api/benchmark_routes.py`**: `AnnotationDetail` gains `fulltext_nougat: str | None` alongside the existing `fulltext` field (kept as-is, still PyMuPDF, for backward compatibility with anything already depending on that name). `_summary`/`AnnotationSummary` gains `has_fulltext_nougat: bool` alongside the existing `has_fulltext`, so the workbench's paper list can show which extraction(s) exist per paper without a second round-trip.

## Frontend design

**New dependencies**: `react-markdown`, `remark-math`, `rehype-katex`, `katex` (+ import `katex/dist/katex.min.css`).

**`app/annotate/[sourceId]/page.tsx`**:
- A small extractor toggle (only shown when both exist for the current paper) — "Nougat" / "PyMuPDF", defaulting to Nougat when `fulltext_nougat` is present, else falling back to the existing plain-text `fulltext`.
- When showing Nougat output: `<ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{detail.fulltext_nougat}</ReactMarkdown>` replaces the `<pre>` block, styled to match the existing serif prose typography.
- When showing PyMuPDF output: unchanged `<pre>` rendering, exactly as today.
- `captureSelection`/`addEvidence` (the existing "select a passage, add as evidence" flow) verified live against the rendered Markdown DOM once implemented — `window.getSelection().toString()` should still work over rendered HTML text nodes, but this gets an explicit check rather than an assumption, per the "verify live, not just with unit tests" practice already established this session.

## Rollout

One-time batch job after implementation: `rb-benchmark-fetch --extractor nougat --force`, re-extracting all 40 benchmark papers with Nougat. Runs in the background (matches this session's pattern for slow ingestion pulls); CPU-only, accepted as potentially slow (minutes per paper × 40 papers). Existing PyMuPDF-extracted `.txt` files are never touched by this run — both outputs coexist per paper afterward, ready for the in-workbench toggle comparison.

**Per-paper failure handling** (`cli_fetch.py`'s existing loop already catches and logs one paper's failure without ending the run — "one unavailable PDF must not end the run" — this extends the same contract to Nougat specifically):
- A Nougat failure for one paper (model error, malformed/corrupt PDF for that pipeline, timeout, etc.) is caught per-paper, logged with the source_id and error detail (same `[i/n] source_id: FAILED <error>` line the loop already prints), and does **not** touch that paper's existing PyMuPDF `.txt` file — it's structurally untouched anyway since the two extractors write to different filenames, but this is now an explicit guarantee, not just a side effect of the file-naming scheme.
- The batch continues to the next paper. The run's final summary (`fetched X, already cached Y, failed Z`) reports Nougat failures the same way PyMuPDF failures are reported today.
- A paper that fails Nougat extraction simply has no `{source_id}.nougat.md` file afterward — the workbench's toggle (which only shows when both extractions exist) naturally falls back to showing PyMuPDF-only for that paper, no special-case UI state needed.

## Amendment (post-implementation): Nougat runs in an isolated subprocess

Implementation of `_extract_nougat` as a direct in-process call (importing `nougat`/`torch` into the main app's environment, as originally specified above) was attempted and reverted. Real live testing found **six independent, unrelated version-drift incompatibilities** between `nougat-ocr==0.1.17` (unmaintained since 2023) and this project's current dependency versions, across three separate fix rounds, with zero successful end-to-end extractions:

1. `transformers` renamed `PretrainedConfig` → `PreTrainedConfig` and moved its module path (transformers 5.x).
2. `albumentations` 2.x rewrote `ImageCompression`'s constructor signature, breaking module-level code `nougat/transforms.py` runs at import time.
3. `transformers` 5.x's `from_pretrained()` finalization now requires `post_init()` to have run; `nougat/model.py`'s `NougatModel.__init__` never calls it.
4. `nougat/dataset/rasterize.py`'s `rasterize_paper` claims (via type hint) to accept raw PDF bytes but its installed implementation only handles `str`/`Path`.
5. `pypdfium2` removed `PdfDocument.render()` (deprecated in 4.25.0 — its own changelog names `nougat` as the motivating example for the deprecation — removed in 5.0.0); the unpinned resolved version (5.13.0) has no such method.
6. After pinning `pypdfium2<5.0.0`, a Windows file-lock cleanup workaround verified against 5.13.0 did not reliably work against the resolved 4.30.0.

This pattern — a new, independent failure at every fix — indicates `nougat-ocr` is not compatible with this project's current dependency graph as a direct import, and patching around each new drift point is not converging. **Revised approach: isolate Nougat in its own subprocess with its own pinned, period-correct dependency environment**, so it never shares (and never fights) the main project's `transformers`/`albumentations`/`pypdfium2` versions.

### Revised backend design

- A separate virtual environment, git-ignored, not part of the main `uv`-managed project — e.g. `.nougat-venv/` at the repo root. Pinned to dependency versions contemporaneous with `nougat-ocr==0.1.17`'s actual 2023 release (check its real upstream `requirements.txt`/`setup.py` from PyPI or GitHub for that release rather than re-guessing versions — the six failures above are exactly the cost of guessing). A one-time, manually-run bootstrap script creates this environment; it is not created automatically on every extraction call (matching the checkpoint download's own "one-time cost" framing already in this spec).
- A standalone extraction script (not part of the `researchbridge` package, since it must run under the isolated interpreter, not the main project's) takes a PDF file path as an argument and writes Markdown output to stdout, doing the real model-loading/rasterization/inference work using the isolated environment's `nougat-ocr`.
- `_extract_nougat(pdf_bytes) -> str` in `fulltext.py` (main project) keeps its exact existing signature and caller contract. Internally: write `pdf_bytes` to a temp file, invoke the isolated environment's Python interpreter as a subprocess running the standalone script against that temp file, capture stdout, return `_tidy(stdout)`. A non-zero subprocess exit raises an exception in the parent — the existing per-paper failure handling in `cli_fetch.py` (a later task, unaffected by this amendment) already catches and logs any exception `extract_text`/`fetch_fulltext` raises, so no new failure-handling code is needed there.
- Everything downstream of `extract_text()` — `fetch_fulltext`, the CLI flags, the API fields, the frontend toggle/rendering — is completely unaffected by this amendment; they only ever consumed `extract_text()`'s string return value, never its internals.

## Testing

- `benchmark/fulltext.py`: unit tests for `fulltext_path`'s extractor-aware filenames, `extract_text`'s dispatch (mocking both extractor implementations so tests don't require `torch`/network), and `fetch_fulltext`'s `force`/cache-skip behavior per extractor.
- `api/benchmark_routes.py`: tests for `fulltext_nougat`/`has_fulltext_nougat` appearing correctly when only one, both, or neither cached file exists.
- Frontend: no test infrastructure exists in this project (established constraint) — verified live in-browser instead, including the text-selection-over-rendered-Markdown check called out above.
- The real Nougat extraction itself (model download, actual inference quality) is verified live against at least one real benchmark PDF during implementation, not mocked — same practice as the Springer/Semantic Scholar connectors' live-verification passes this session.
