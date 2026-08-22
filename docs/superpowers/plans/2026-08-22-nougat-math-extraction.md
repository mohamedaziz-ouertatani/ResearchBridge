# Nougat Math-Aware Benchmark Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Nougat as a second, selectable full-text extractor for the 40-paper benchmark slice, alongside the existing PyMuPDF extractor (never replacing it), so math-heavy papers can be read with real typeset equations in the annotation workbench and the two extraction methods can be compared.

**Architecture:** `benchmark/fulltext.py`'s `extract_text()` becomes a dispatcher over two implementations (`_extract_pymupdf`, unchanged behavior; `_extract_nougat`, new) selected by an `extractor` parameter. Cached output goes to per-extractor filenames (`{source_id}.txt` for PyMuPDF, `{source_id}.nougat.md` for Nougat) so re-running one never overwrites the other. The API exposes both cached texts per paper; the workbench renders Nougat's Markdown+LaTeX with `react-markdown`+KaTeX and falls back to the existing plain-text `<pre>` render for PyMuPDF or when Nougat has no cached output.

**Tech Stack:** Python (FastAPI, SQLAlchemy already in use), `nougat-ocr` (new — pulls in `torch`), Next.js/React frontend, `react-markdown` + `remark-math` + `rehype-katex` + `katex` (new).

**Spec:** `docs/superpowers/specs/2026-08-22-nougat-math-extraction-design.md`

## Global Constraints

- PyMuPDF's existing extraction behavior and the 40 already-cached `{source_id}.txt` files must never be modified or overwritten by this work.
- `nougat-ocr`'s actual installed API (checkpoint selection, model-loading call, download/caching behavior) must be verified live against the real installed package — do not assume a specific function signature or checkpoint name from memory; the library's public API has changed across releases.
- A Nougat failure on one paper must not abort a batch run: log the error with the source_id, leave that paper's PyMuPDF output untouched, and continue to the next paper.
- No change to the main ingestion pipeline (arXiv/Springer/Semantic Scholar connectors, `IngestionPipeline`) — this plan touches only `benchmark/`, `api/benchmark_routes.py`, and the `app/annotate/[sourceId]` frontend page.
- This project has no frontend test infrastructure — frontend changes are verified live in the browser, not with automated tests.

---

## Task 1: Extractor-aware `fulltext_path`

**Files:**
- Modify: `src/researchbridge/benchmark/fulltext.py`
- Test: `tests/test_benchmark_fulltext.py`

**Interfaces:**
- Produces: `fulltext_path(output_dir: Path, source_id: str, extractor: str = "pymupdf") -> Path`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark_fulltext.py`:

```python
from researchbridge.benchmark.fulltext import fulltext_path


def test_fulltext_path_defaults_to_pymupdf_txt_filename(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678") == tmp_path / "1234.5678.txt"


def test_fulltext_path_pymupdf_extractor_matches_default(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678", extractor="pymupdf") == tmp_path / "1234.5678.txt"


def test_fulltext_path_nougat_extractor_uses_markdown_filename(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678", extractor="nougat") == tmp_path / "1234.5678.nougat.md"


def test_fulltext_path_rejects_unknown_extractor(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="extractor"):
        fulltext_path(tmp_path, "1234.5678", extractor="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_fulltext.py -k fulltext_path -v`
Expected: FAIL — `fulltext_path() got an unexpected keyword argument 'extractor'`

- [ ] **Step 3: Implement**

In `src/researchbridge/benchmark/fulltext.py`, replace the existing `fulltext_path`:

```python
_EXTRACTOR_FILENAMES = {
    "pymupdf": "{source_id}.txt",
    "nougat": "{source_id}.nougat.md",
}


def fulltext_path(output_dir: Path, source_id: str, extractor: str = "pymupdf") -> Path:
    if extractor not in _EXTRACTOR_FILENAMES:
        raise ValueError(f"extractor must be one of {sorted(_EXTRACTOR_FILENAMES)}, got {extractor!r}")
    return output_dir / _EXTRACTOR_FILENAMES[extractor].format(source_id=source_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_fulltext.py -v`
Expected: all PASS (including the pre-existing `_tidy` tests, unaffected)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/benchmark/fulltext.py tests/test_benchmark_fulltext.py
git commit -m "feat: make fulltext_path extractor-aware (pymupdf/nougat)"
```

---

## Task 2: `extract_text` dispatches by extractor (PyMuPDF path behavior-preserving)

**Files:**
- Modify: `src/researchbridge/benchmark/fulltext.py`
- Test: `tests/test_benchmark_fulltext.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `extract_text(pdf_bytes: bytes, extractor: str = "pymupdf") -> str`; internal `_extract_pymupdf(pdf_bytes: bytes) -> str` (today's implementation, moved verbatim — same PyMuPDF calls, same `_tidy()` call at the end)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark_fulltext.py`:

```python
def test_extract_text_dispatches_to_pymupdf_by_default(monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    monkeypatch.setattr(ft, "_extract_pymupdf", lambda pdf_bytes: "pymupdf output")

    assert ft.extract_text(b"fake-pdf-bytes") == "pymupdf output"


def test_extract_text_dispatches_to_nougat_when_selected(monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    monkeypatch.setattr(ft, "_extract_nougat", lambda pdf_bytes: "nougat output")

    assert ft.extract_text(b"fake-pdf-bytes", extractor="nougat") == "nougat output"


def test_extract_text_rejects_unknown_extractor() -> None:
    import pytest

    from researchbridge.benchmark.fulltext import extract_text

    with pytest.raises(ValueError, match="extractor"):
        extract_text(b"fake-pdf-bytes", extractor="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_fulltext.py -k extract_text_dispatches -v`
Expected: FAIL — `extract_text() got an unexpected keyword argument 'extractor'` (and `_extract_pymupdf`/`_extract_nougat` don't exist yet)

- [ ] **Step 3: Implement**

In `src/researchbridge/benchmark/fulltext.py`:

1. Rename the current `extract_text(pdf_bytes: bytes) -> str` function to `_extract_pymupdf(pdf_bytes: bytes) -> str` — body unchanged (the `import pymupdf` lazy-import, the `pymupdf.open(...)` call, the `_tidy("\n\n".join(pages))` return, all stay exactly as they are today).
2. Add a new `extract_text` that dispatches:

```python
def extract_text(pdf_bytes: bytes, extractor: str = "pymupdf") -> str:
    """Extract readable text from a PDF using the given extractor.

    "pymupdf": fast, always available, loses math notation (see module
    docstring). "nougat": math-aware, much slower, a real ML model - see
    _extract_nougat's docstring for what its output actually looks like.
    """
    if extractor == "pymupdf":
        return _extract_pymupdf(pdf_bytes)
    if extractor == "nougat":
        return _extract_nougat(pdf_bytes)
    raise ValueError(f"extractor must be one of ['nougat', 'pymupdf'], got {extractor!r}")
```

3. Leave a temporary `_extract_nougat` stub so the module imports cleanly (Task 3 replaces this):

```python
def _extract_nougat(pdf_bytes: bytes) -> str:
    raise NotImplementedError("implemented in Task 3")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_fulltext.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/benchmark/fulltext.py tests/test_benchmark_fulltext.py
git commit -m "feat: extract_text dispatches by extractor, pymupdf path unchanged"
```

---

## Task 3: Investigate the installed `nougat-ocr` API and implement `_extract_nougat`

This task is different from the others: the exact model-loading call cannot be written from memory (per Global Constraints) — it must be discovered by installing and inspecting the real package first. Steps 1-2 are investigation; steps 3+ implement based on what's actually found.

**Files:**
- Modify: `pyproject.toml` (add dependency)
- Modify: `src/researchbridge/benchmark/fulltext.py`
- Test: `tests/test_benchmark_fulltext.py`

**Interfaces:**
- Consumes: `_tidy(text: str) -> str` (existing)
- Produces: `_extract_nougat(pdf_bytes: bytes) -> str` — real implementation, replacing Task 2's stub

- [ ] **Step 1: Install the package**

```bash
uv add nougat-ocr
```

Record the exact version installed (check `uv.lock` or `uv pip show nougat-ocr`) — write it into the module docstring in Step 4 below.

- [ ] **Step 2: Inspect the installed package's real public API**

Run each of these and read the output before writing any implementation code:

```bash
uv run python -c "import nougat; print(nougat.__file__)"
uv run python -c "import nougat; help(nougat)"
```

Also read the actual source under the path the first command prints (e.g. `.venv/Lib/site-packages/nougat/`), specifically looking for:
- The model class/loading function (commonly `NougatModel` with a `.from_pretrained(...)` classmethod, but confirm against the real source — do not assume).
- How it expects PDF input (raw bytes? a file path? pre-rasterized page images via `pypdf`/`pdf2image`?).
- The checkpoint identifier its own default points to (e.g. via a `get_checkpoint(...)` helper or a hardcoded default in its CLI module `nougat/cli.py` if present).
- Whether it exposes a simple importable inference function, or expects you to drive it the way `nougat/cli.py`'s `main()` does (in which case, follow that same sequence from your own code rather than re-inventing it).

- [ ] **Step 3: Write a failing smoke-test-shaped test using what you found**

This test mocks the actual model call you identified in Step 2 (name it precisely — replace `<ModelClass>`/`<load_method>`/`<predict_method>` below with what you actually found):

```python
def test_extract_nougat_returns_the_models_markdown_output(monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    class FakeModel:
        def <predict_method>(self, *args, **kwargs):
            return "# Title\n\nSome text with $\\alpha_i$ math."

    monkeypatch.setattr(ft, "_load_nougat_model", lambda: FakeModel())

    result = ft._extract_nougat(b"fake-pdf-bytes")

    assert "alpha_i" in result or "\\alpha_i" in result
```

Adjust the fake to match the real call shape found in Step 2 (it may need to mock a page-rasterization step too, in which case add that mock alongside `_load_nougat_model`).

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark_fulltext.py -k extract_nougat -v`
Expected: FAIL — `_load_nougat_model` (or whichever helper you introduced) doesn't exist yet

- [ ] **Step 5: Implement `_extract_nougat` using the real, verified API**

In `src/researchbridge/benchmark/fulltext.py`, replace Task 2's stub. The shape below is a skeleton — fill in the actual model-loading and inference calls from Step 2's findings, and update the docstring with the real version/checkpoint/behavior you observed (mirroring how `springer.py`/`semantic_scholar.py` document their own live-verified gotchas elsewhere in this codebase):

```python
def _extract_nougat(pdf_bytes: bytes) -> str:
    """Math-aware extraction via Nougat (nougat-ocr==<VERSION>).

    <Fill in during implementation: what checkpoint this loads and why,
    how PDF bytes get turned into model input, and any real output
    quirks observed against a real benchmark PDF in Task 4 - the same
    way springer.py/semantic_scholar.py document their own live-
    verified behavior.>
    """
    model = _load_nougat_model()
    # <the real inference call, from Step 2's findings>
    markdown = ...
    return _tidy(markdown)


def _load_nougat_model():
    """Lazily imports and loads the Nougat model - kept as its own function
    so tests can monkeypatch model loading without a real multi-GB download."""
    <the real import + load call from Step 2's findings>
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark_fulltext.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/researchbridge/benchmark/fulltext.py tests/test_benchmark_fulltext.py
git commit -m "feat: implement Nougat extraction (nougat-ocr)"
```

---

## Task 4: Live smoke-test against a real benchmark PDF, verify `_tidy()` is Markdown-safe

**Files:**
- Modify: `src/researchbridge/benchmark/fulltext.py` (only if Step 3 below finds `_tidy` mangles Markdown)

- [ ] **Step 1: Run a real extraction against a real cached PDF**

Use the paper already referenced in the design spec (`1812.02641`, "Local Conditioning in Undirected Networks") or any other paper with a `.txt` already cached under `benchmark/fulltext/`. Since only the PyMuPDF text is cached (not the original PDF bytes), re-download the PDF first:

```bash
uv run python -c "
import requests
from researchbridge.benchmark.fulltext import extract_text
r = requests.get('https://arxiv.org/pdf/1812.02641', timeout=60)
r.raise_for_status()
text = extract_text(r.content, extractor='nougat')
print(text[:3000])
print('---')
print('total length:', len(text))
"
```

This will trigger the model's first-run checkpoint download — expect this to take a while and print progress; let it finish.

- [ ] **Step 2: Compare against the known-garbled PyMuPDF output**

Check that the math expressions that were scrambled in `benchmark/fulltext/1812.02641.txt` (the `Φi = eαi / e−αi` matrix, the `∑`/`∏` operators that came out as bare `X`/`Y`) now appear as recognizable LaTeX (e.g. `$\Phi_i$`, `\sum`, `\prod`) in the Nougat output. If the output looks wrong (garbled differently, empty, or clearly not math-aware), stop and re-check Task 3's Step 2 findings before proceeding — do not paper over a broken integration.

- [ ] **Step 3: Check `_tidy()` against the real Markdown output**

Inspect whether `_tidy()`'s blank-line-collapsing regex (`re.sub(r"\n{3,}", "\n\n", text)`) altered any Markdown structure incorrectly (e.g. collapsed a blank line that separated two list items or a heading from its paragraph in a way that changes rendered meaning). Markdown only needs a single blank line between block elements, so 3+ blank lines collapsing to 2 should be harmless — but confirm against the real output from Step 1 rather than assuming.

If a real problem is found, fix `_tidy()` and re-run Task 3's tests to confirm nothing there regressed. If nothing is wrong, no code change needed for this task.

- [ ] **Step 4: Commit (only if Step 3 required a fix)**

```bash
git add src/researchbridge/benchmark/fulltext.py
git commit -m "fix: adjust _tidy for Markdown output from Nougat"
```

---

## Task 5: `fetch_fulltext` gains `extractor` and `force` parameters

**Files:**
- Modify: `src/researchbridge/benchmark/fulltext.py`
- Test: `tests/test_benchmark_fulltext.py`

**Interfaces:**
- Consumes: `fulltext_path(output_dir, source_id, extractor="pymupdf")` (Task 1), `extract_text(pdf_bytes, extractor="pymupdf")` (Task 2)
- Produces: `fetch_fulltext(source_id: str, output_dir: Path, session: requests.Session | None = None, extractor: str = "pymupdf", force: bool = False) -> str`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark_fulltext.py`:

```python
def test_fetch_fulltext_uses_extractor_specific_cache_path(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    (tmp_path / "1234.5678.nougat.md").write_text("cached nougat text", encoding="utf-8")

    result = ft.fetch_fulltext("1234.5678", tmp_path, extractor="nougat")

    assert result == "cached nougat text"


def test_fetch_fulltext_force_bypasses_the_cache(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    (tmp_path / "1234.5678.txt").write_text("stale cached text", encoding="utf-8")

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": "freshly extracted text")

    result = ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), force=True)

    assert result == "freshly extracted text"
    assert (tmp_path / "1234.5678.txt").read_text(encoding="utf-8") == "freshly extracted text"


def test_fetch_fulltext_passes_extractor_through_to_extract_text(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    calls = []
    monkeypatch.setattr(
        ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": calls.append(extractor) or "text"
    )

    ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), extractor="nougat")

    assert calls == ["nougat"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_fulltext.py -k fetch_fulltext -v`
Expected: FAIL — `fetch_fulltext() got an unexpected keyword argument 'extractor'`

- [ ] **Step 3: Implement**

Replace `fetch_fulltext` in `src/researchbridge/benchmark/fulltext.py`:

```python
def fetch_fulltext(
    source_id: str,
    output_dir: Path,
    session: requests.Session | None = None,
    extractor: str = "pymupdf",
    force: bool = False,
) -> str:
    """Download one paper's PDF and cache its extracted text. Returns the text.

    force=True bypasses the cache-hit check and re-extracts even if a
    cached file already exists for this (source_id, extractor) pair -
    needed to deliberately re-run with a different/updated extractor
    rather than silently keeping a stale cached result.
    """
    path = fulltext_path(output_dir, source_id, extractor=extractor)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    http = session or requests
    response = http.get(
        ARXIV_PDF_URL.format(source_id=source_id),
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "ResearchBridge/0.1 (benchmark annotation; solo research project)"},
    )
    response.raise_for_status()

    text = extract_text(response.content, extractor=extractor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_fulltext.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/benchmark/fulltext.py tests/test_benchmark_fulltext.py
git commit -m "feat: fetch_fulltext supports extractor selection and force re-extraction"
```

---

## Task 6: `rb-benchmark-fetch` gains `--extractor`/`--force`, with per-paper Nougat failure handling

**Files:**
- Modify: `src/researchbridge/benchmark/cli_fetch.py`

**Interfaces:**
- Consumes: `fetch_fulltext(source_id, output_dir, session, extractor, force)` (Task 5), `fulltext_path(output_dir, source_id, extractor)` (Task 1)

- [ ] **Step 1: Implement**

Replace the body of `src/researchbridge/benchmark/cli_fetch.py`:

```python
def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Fetch full text for the benchmark papers.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--extractor", choices=["pymupdf", "nougat"], default="pymupdf",
        help="Which extraction engine to use (default: pymupdf).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if a cached file already exists for this extractor.",
    )
    args = parser.parse_args()

    output_dir = args.benchmark_dir / "fulltext"
    annotations = load_all(args.benchmark_dir)
    if not annotations:
        print(f"No annotation files in {args.benchmark_dir}/annotations/ - run rb-benchmark-sample first.")
        return

    fetched = skipped = 0
    failures: list[tuple[str, str]] = []
    http = requests.Session()

    for i, annotation in enumerate(annotations, start=1):
        source_id = annotation.source_id
        if not args.force and fulltext_path(output_dir, source_id, extractor=args.extractor).exists():
            skipped += 1
            continue

        if fetched > 0:
            throttle()

        try:
            text = fetch_fulltext(
                source_id, output_dir, session=http, extractor=args.extractor, force=args.force
            )
        except Exception as exc:  # one unavailable PDF (or one failed Nougat run) must not end the run
            failures.append((source_id, str(exc)[:200]))
            print(f"[{i}/{len(annotations)}] {source_id}: FAILED {exc}", flush=True)
            continue

        fetched += 1
        print(f"[{i}/{len(annotations)}] {source_id}: {len(text.split()):,} words", flush=True)

    print(f"\nfetched {fetched}, already cached {skipped}, failed {len(failures)}")
    for source_id, detail in failures:
        print(f"  {source_id}: {detail}")
```

Note: `fetch_fulltext` already writes to the extractor-specific path (Task 1/5), so a failed Nougat extraction never touches the existing PyMuPDF `.txt` file for that paper — they're different files by construction, and the exception is caught before any write happens for the failing paper.

- [ ] **Step 2: Verify manually**

```bash
uv run python -m researchbridge.benchmark.cli_fetch --extractor pymupdf --benchmark-dir benchmark
```

Expected: all 40 papers report "already cached" (`skipped`), since PyMuPDF's default behavior and cache files are unchanged — this confirms the refactor didn't touch existing behavior. Do NOT run `--extractor nougat --force` yet — that's Task 9 (rollout), after the frontend is done.

- [ ] **Step 3: Commit**

```bash
git add src/researchbridge/benchmark/cli_fetch.py
git commit -m "feat: rb-benchmark-fetch supports --extractor and --force"
```

---

## Task 7: API exposes both cached extractions per paper

**Files:**
- Modify: `src/researchbridge/api/benchmark_routes.py`
- Test: `tests/test_benchmark_api.py`

**Interfaces:**
- Consumes: `fulltext_path(output_dir, source_id, extractor)` (Task 1)
- Produces: `AnnotationSummary.has_fulltext_nougat: bool`, `AnnotationDetail.fulltext_nougat: str | None`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_benchmark_api.py`. First extend the `benchmark_dir` fixture to also cache a Nougat file for paper 1:

```python
@pytest.fixture()
def benchmark_dir(tmp_path) -> Path:
    annotations = tmp_path / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "arxiv_1111.11111.yaml").write_text(_template("1111.11111"), encoding="utf-8")
    (annotations / "arxiv_2222.22222.yaml").write_text(
        _template("2222.22222", domain="Systems"), encoding="utf-8"
    )

    fulltext = tmp_path / "fulltext"
    fulltext.mkdir()
    (fulltext / "1111.11111.txt").write_text("Full text of the first paper.", encoding="utf-8")
    (fulltext / "1111.11111.nougat.md").write_text("# Full text\n\nWith $\\alpha$ math.", encoding="utf-8")
    return tmp_path
```

(This replaces the existing fixture in-place — same paper ids, one added file.)

Then add:

```python
def test_list_reports_which_papers_have_nougat_fulltext(client) -> None:
    by_id = {row["source_id"]: row for row in client.get("/api/benchmark/papers").json()}

    assert by_id["1111.11111"]["has_fulltext_nougat"] is True
    assert by_id["2222.22222"]["has_fulltext_nougat"] is False


def test_detail_includes_nougat_fulltext_when_cached(client) -> None:
    body = client.get("/api/benchmark/papers/1111.11111").json()

    assert body["fulltext_nougat"] == "# Full text\n\nWith $\\alpha$ math."


def test_detail_nougat_fulltext_is_null_when_not_cached(client) -> None:
    assert client.get("/api/benchmark/papers/2222.22222").json()["fulltext_nougat"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_benchmark_api.py -k nougat -v`
Expected: FAIL — `KeyError: 'has_fulltext_nougat'` / `KeyError: 'fulltext_nougat'`

- [ ] **Step 3: Implement**

In `src/researchbridge/api/benchmark_routes.py`:

1. Add the import: `from researchbridge.benchmark.fulltext import fulltext_path` is already imported — no change needed there.
2. Extend `AnnotationSummary`:

```python
class AnnotationSummary(BaseModel):
    source_id: str
    title: str | None
    domain: str | None
    year: int | None
    filled: int
    total: int
    is_complete: bool
    has_fulltext: bool
    has_fulltext_nougat: bool
```

3. Extend `AnnotationDetail`:

```python
class AnnotationDetail(AnnotationSummary):
    url: str | None
    fields: dict[str, str]
    research_gap: dict[str, str]
    key_evidence: list[EvidenceItem]
    fulltext: str | None
    fulltext_nougat: str | None
```

4. Update `_summary`:

```python
def _summary(annotation: Annotation, benchmark_dir: Path) -> AnnotationSummary:
    identity = annotation.identity
    year = identity.get("year")
    fulltext_dir = benchmark_dir / "fulltext"
    return AnnotationSummary(
        source_id=annotation.source_id,
        title=identity.get("title"),
        domain=identity.get("domain"),
        year=int(year) if year else None,
        filled=annotation.filled_count,
        total=annotation.total_count,
        is_complete=annotation.is_complete,
        has_fulltext=fulltext_path(fulltext_dir, annotation.source_id).exists(),
        has_fulltext_nougat=fulltext_path(fulltext_dir, annotation.source_id, extractor="nougat").exists(),
    )
```

5. Update `get_annotation`:

```python
@router.get("/papers/{source_id}", response_model=AnnotationDetail)
def get_annotation(
    source_id: str, benchmark_dir: Path = Depends(get_benchmark_dir)
) -> AnnotationDetail:
    annotation = _find(source_id, benchmark_dir)
    fulltext_dir = benchmark_dir / "fulltext"
    path = fulltext_path(fulltext_dir, source_id)
    nougat_path = fulltext_path(fulltext_dir, source_id, extractor="nougat")

    return AnnotationDetail(
        **_summary(annotation, benchmark_dir).model_dump(),
        url=annotation.identity.get("url"),
        fields={name: annotation.fields.get(name, "") for name in ANNOTATION_FIELDS},
        research_gap=annotation.research_gap,
        key_evidence=[EvidenceItem(**item) for item in annotation.key_evidence],
        fulltext=path.read_text(encoding="utf-8") if path.exists() else None,
        fulltext_nougat=nougat_path.read_text(encoding="utf-8") if nougat_path.exists() else None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benchmark_api.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/api/benchmark_routes.py tests/test_benchmark_api.py
git commit -m "feat: expose Nougat fulltext alongside PyMuPDF in the benchmark API"
```

---

## Task 8: Frontend dependencies and API types

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/lib/benchmarkApi.ts`

**Interfaces:**
- Produces: `AnnotationSummary.has_fulltext_nougat: boolean`, `AnnotationDetail.fulltext_nougat: string | null` (matching Task 7's response shape)

- [ ] **Step 1: Install dependencies**

```bash
cd frontend
npm install react-markdown remark-math rehype-katex katex
```

- [ ] **Step 2: Update types**

In `frontend/lib/benchmarkApi.ts`:

```typescript
export type AnnotationSummary = {
  source_id: string;
  title: string | null;
  domain: string | null;
  year: number | null;
  filled: number;
  total: number;
  is_complete: boolean;
  has_fulltext: boolean;
  has_fulltext_nougat: boolean;
};

export type AnnotationDetail = AnnotationSummary & {
  url: string | null;
  fields: Record<string, string>;
  research_gap: { addressed: string; remaining: string };
  key_evidence: EvidenceItem[];
  fulltext: string | null;
  fulltext_nougat: string | null;
};
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors (nothing consumes the new fields yet, so nothing should break; if `AnnotationDetail`'s consumers show errors, they're addressed in Task 9)

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/lib/benchmarkApi.ts
git commit -m "feat: add react-markdown/katex deps and Nougat fields to benchmark API types"
```

---

## Task 9: Workbench renders Nougat Markdown+math with a PyMuPDF/Nougat toggle

**Files:**
- Modify: `frontend/app/annotate/[sourceId]/page.tsx`

**Interfaces:**
- Consumes: `AnnotationDetail.fulltext`, `AnnotationDetail.fulltext_nougat` (Task 8)

- [ ] **Step 1: Implement**

In `frontend/app/annotate/[sourceId]/page.tsx`:

1. Add imports at the top:

```typescript
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
```

2. Add extractor-selection state, right after the existing `useState` declarations in `Workbench`:

```typescript
const [extractorView, setExtractorView] = useState<"nougat" | "pymupdf">("nougat");
```

3. Reset it alongside the other per-paper state in the `useEffect` that loads paper detail (the one calling `benchmarkApi.detail(sourceId)`), so switching papers doesn't carry over the toggle choice from a previous paper:

```typescript
useEffect(() => {
    dirty.current = false;
    setSaveState("idle");
    setSelection("");
    setExtractorView("nougat");
    // ... existing benchmarkApi.detail(...) call below, unchanged
```

4. Replace the paper-body render block (the `{detail && !detail.fulltext ? ... : <pre>...</pre>}` section) with:

```tsx
{detail && !detail.fulltext && !detail.fulltext_nougat ? (
  <p className="mt-6 border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] text-[var(--ink-soft)]">
    No full text cached for this paper. Run{" "}
    <code className="readout text-[0.875rem]">rb-benchmark-fetch</code> to download it.
  </p>
) : (
  <>
    {detail?.fulltext_nougat && detail?.fulltext && (
      <div className="mb-3 flex gap-1">
        <button
          onClick={() => setExtractorView("nougat")}
          className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
            extractorView === "nougat"
              ? "border-[var(--ink)] text-[var(--ink)]"
              : "border-[var(--rule)] text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
          }`}
        >
          Nougat
        </button>
        <button
          onClick={() => setExtractorView("pymupdf")}
          className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
            extractorView === "pymupdf"
              ? "border-[var(--ink)] text-[var(--ink)]"
              : "border-[var(--rule)] text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
          }`}
        >
          PyMuPDF
        </button>
      </div>
    )}

    {(extractorView === "nougat" ? detail?.fulltext_nougat : detail?.fulltext) ? (
      extractorView === "nougat" && detail?.fulltext_nougat ? (
        <div className="mt-5 max-w-none font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] text-[var(--ink)]">
          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
            {detail.fulltext_nougat}
          </ReactMarkdown>
        </div>
      ) : (
        <pre className="mt-5 font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] whitespace-pre-wrap text-[var(--ink)]">
          {detail?.fulltext}
        </pre>
      )
    ) : (
      <pre className="mt-5 font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] whitespace-pre-wrap text-[var(--ink)]">
        {detail?.fulltext}
      </pre>
    )}
  </>
)}
```

(The toggle only renders when both texts exist, per the spec — a paper with only PyMuPDF cached shows the plain `<pre>` exactly as it does today, no toggle, no behavior change for the 39 other papers until Task 10's batch job runs.)

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Verify live**

Start the frontend and API dev servers, open `/annotate/1812.02641` (or whichever paper Task 4's smoke test used, assuming its Nougat output was also written to the real `benchmark/fulltext/` directory — if not, temporarily copy Task 4's smoke-test output to `benchmark/fulltext/1812.02641.nougat.md` for this check). Confirm:
- The toggle appears and defaults to "Nougat".
- Nougat's math renders as real typeset equations (not raw `$...$` text).
- Switching to "PyMuPDF" shows the original plain-text render, unchanged.
- A paper with no Nougat cache (any other benchmark paper right now) shows no toggle and renders exactly as before.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/annotate/[sourceId]/page.tsx
git commit -m "feat: render Nougat Markdown+math with a PyMuPDF/Nougat toggle in the workbench"
```

---

## Task 10: Verify text-selection evidence capture against the rendered Markdown DOM

**Files:**
- Modify: `frontend/app/annotate/[sourceId]/page.tsx` (only if Step 1 finds a real problem)

- [ ] **Step 1: Test selection live**

With the same paper from Task 9's Step 3 open and the Nougat view active, select a plain-text passage (not inside a math expression) with the mouse, confirm the "add selection as evidence" button appears (the existing `captureSelection`/`onMouseUp` handler), click it, and confirm the passage is added to the evidence list correctly (matching today's behavior for the plain-text `<pre>` case).

Also try selecting text that spans across a rendered KaTeX element (e.g. a sentence containing an inline equation) and confirm `window.getSelection()?.toString()` still produces reasonable text (KaTeX's internal DOM structure can sometimes fragment selected text oddly — check this specifically rather than assuming it's fine).

- [ ] **Step 2: Fix if needed**

If selection across math is broken (e.g. produces empty string or garbled duplicated text), the fix is scoped narrowly: KaTeX renders both a visually-hidden MathML representation and a visible HTML representation for accessibility, which can cause `getSelection()` to pick up duplicated text. If this happens, wrap the `ReactMarkdown` output in a container with `className="katex-html-only"` and add to `frontend/app/globals.css`:

```css
.katex-html-only .katex-mathml {
  display: none;
}
```

This hides the (usually redundant) MathML copy from both rendering and selection, leaving only KaTeX's visible HTML representation selectable.

- [ ] **Step 3: Commit (only if Step 2 required a fix)**

```bash
git add frontend/app/annotate/[sourceId]/page.tsx frontend/app/globals.css
git commit -m "fix: prevent duplicated text selection across KaTeX-rendered math"
```

---

## Task 11: Rollout — re-extract all 40 benchmark papers with Nougat

**Files:** none (operational task)

- [ ] **Step 1: Run the batch job in the background**

```bash
uv run python -m researchbridge.benchmark.cli_fetch --extractor nougat --force --benchmark-dir benchmark
```

This is CPU-only and expected to be slow (potentially minutes per paper × 40 papers) — run it in the background and check progress periodically rather than blocking on it.

- [ ] **Step 2: Review the summary**

Once finished, check the printed `fetched X, already cached Y, failed Z` line and the per-failure detail lines. For any failures, confirm (per Task 6's contract) that the corresponding paper's original `{source_id}.txt` (PyMuPDF) file is still present and unchanged:

```bash
ls benchmark/fulltext/*.txt | wc -l   # should still be 40 (or however many existed before this task)
```

- [ ] **Step 3: Spot-check a few papers in the workbench**

Open 2-3 papers in `/annotate/{source_id}` (mix of ones that succeeded and, if any, ones that failed) and confirm the toggle/rendering behaves as expected in each case.

- [ ] **Step 4: Report results**

No commit for this task (it produces data files under `benchmark/fulltext/`, not code) — report the final fetched/cached/failed counts.

---

## Self-Review Notes

**Spec coverage:**
- Dual extractor behind `extract_text()`, PyMuPDF untouched → Tasks 1, 2, 5
- Per-extractor cached filenames → Task 1
- `--extractor`/`--force` CLI flags → Task 6
- Nougat checkpoint/API verified live, not assumed → Task 3 (investigation-first structure)
- Per-paper failure handling in the batch job → Task 6 (existing try/except contract extended and documented), Task 11 (verified against real run)
- API exposes both cached texts → Task 7
- Frontend renders Markdown+math with a toggle → Task 9
- Text-selection verified against rendered DOM → Task 10
- Batch re-extraction rollout → Task 11

**Placeholder scan:** Task 3 intentionally contains a partially-filled skeleton (`_extract_nougat`'s body, `_load_nougat_model`) rather than concrete code — this is not an oversight; the spec's approved review feedback explicitly requires the checkpoint/model-loading call to be discovered from the real installed package rather than assumed from memory, and Task 3's steps 1-2 are concrete, runnable investigation commands that produce the information Step 5 needs. Every other task has fully concrete code.

**Type consistency:** `extractor: str = "pymupdf"` and the two-value set `{"pymupdf", "nougat"}` are used identically across `fulltext_path` (Task 1), `extract_text` (Task 2), and `fetch_fulltext` (Task 5). `has_fulltext_nougat`/`fulltext_nougat` naming is consistent between the backend schema (Task 7) and frontend types (Task 8).
