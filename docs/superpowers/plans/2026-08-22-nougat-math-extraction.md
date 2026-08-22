# Nougat Math-Aware Benchmark Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Nougat as a second, selectable full-text extractor for the 40-paper benchmark slice, alongside the existing PyMuPDF extractor (never replacing it), so math-heavy papers can be read with real typeset equations in the annotation workbench and the two extraction methods can be compared.

**Architecture:** `benchmark/fulltext.py`'s `extract_text()` becomes a dispatcher over two implementations (`_extract_pymupdf`, unchanged behavior; `_extract_nougat`, new) selected by an `extractor` parameter. Cached output goes to per-extractor filenames (`{source_id}.txt` for PyMuPDF, `{source_id}.nougat.md` for Nougat) so re-running one never overwrites the other. The API exposes both cached texts per paper; the workbench renders Nougat's Markdown+LaTeX with `react-markdown`+KaTeX and falls back to the existing plain-text `<pre>` render for PyMuPDF or when Nougat has no cached output.

**AMENDMENT (see spec's "Amendment" section):** `_extract_nougat` runs Nougat in an isolated subprocess with its own pinned, period-correct dependency environment, rather than importing `nougat`/`torch` directly into the main project's environment. Six independent, unrelated dependency-version incompatibilities were found attempting the direct-import approach (documented in the spec and this plan's Task 3/4 history) — the subprocess isolation avoids fighting the main project's modern `transformers`/`albumentations`/`pypdfium2` versions entirely. Tasks 1-2 (already complete) and Tasks 5-11 (unaffected — they only ever consumed `extract_text()`'s string return value) are unchanged; Tasks 3-4 below are the amended versions.

**Tech Stack:** Python (FastAPI, SQLAlchemy already in use), a separate isolated venv for `nougat-ocr` (pinned to period-correct dependency versions, NOT part of the main `uv` project), Next.js/React frontend, `react-markdown` + `remark-math` + `rehype-katex` + `katex` (new).

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

## Task 3 (AMENDED): Set up an isolated Nougat subprocess environment and implement `_extract_nougat`

**This task replaces the original in-process Task 3.** The original approach (importing `nougat`/`torch` directly into the main project's environment) was attempted and reverted after six independent, unrelated dependency-version incompatibilities across three fix rounds — see the spec's "Amendment" section for the full history. This version isolates Nougat in its own subprocess with its own pinned, period-correct dependency environment, so it never shares (and never fights) the main project's `transformers`/`albumentations`/`pypdfium2` versions.

This task is still genuinely investigative for one specific reason: the *correct pinned dependency versions* for the isolated environment must be discovered from `nougat-ocr==0.1.17`'s actual real-world release context (its own declared/tested requirements at release time, circa 2023), not guessed — guessing is exactly what caused the six failures being amended here.

**Files:**
- Create: `.nougat-venv/` (git-ignored isolated virtual environment — not tracked in git, but its *setup* is scripted and committed)
- Create: `scripts/setup_nougat_env.sh` (or `.ps1` if more natural on this Windows machine — your call; a one-time, manually-run bootstrap, not run automatically by the app)
- Create: `scripts/nougat_extract.py` (standalone script, runs under the isolated venv's interpreter — NOT part of the `researchbridge` package/import path, since it must be importable/runnable only by the isolated Python, not the main project's)
- Modify: `src/researchbridge/benchmark/fulltext.py`
- Test: `tests/test_benchmark_fulltext.py`

**Interfaces:**
- Consumes: `_tidy(text: str) -> str` (existing)
- Produces: `_extract_nougat(pdf_bytes: bytes) -> str` — real implementation, replacing Task 2's stub. Same signature and caller contract as originally specified — this is a pure internal-implementation change.

- [ ] **Step 1: Determine the correct pinned dependency versions**

Check `nougat-ocr==0.1.17`'s actual declared dependencies at its real PyPI/GitHub release (its own `setup.py`/`requirements.txt`/`pyproject.toml` from the facebookresearch/nougat repository, or PyPI's release metadata for that exact version and its release date). Cross-reference against what's already confirmed to work from this session's investigation:
- `albumentations<2.0.0` (confirmed working, e.g. `1.4.24`)
- A `transformers` version old enough to predate the `post_init()`-required-in-`from_pretrained` change (the exact version that introduced this requirement was not bisected this session — check `transformers`' own changelog/git blame for `_finalize_model_loading`/`all_tied_weights_keys`, or simply use whatever `nougat-ocr==0.1.17` itself was tested against circa 2023, likely a `4.x` version)
- A `pypdfium2` version with `PdfDocument.render()` intact — confirmed deprecated in `4.25.0`, removed in `5.0.0`; pin below `4.25.0` to avoid even the deprecation-transition period, unless nougat's own declared requirement says otherwise

Record the exact versions you land on and why, in the setup script's comments.

- [ ] **Step 2: Write the environment bootstrap script**

`scripts/setup_nougat_env.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# One-time setup for the isolated Nougat extraction environment. Not run
# automatically - Nougat (nougat-ocr, last released 2023) is incompatible
# with this project's main environment's transformers/albumentations/
# pypdfium2 versions (see docs/superpowers/specs/2026-08-22-nougat-math-
# extraction-design.md's Amendment section for the six failures found
# attempting a direct in-process import). This isolated venv is pinned to
# period-correct versions instead.

python -m venv .nougat-venv
# Activate per-platform (this repo runs on Windows - adjust activation for
# the shell you actually test in; document both if needed)
.nougat-venv/Scripts/pip install --upgrade pip
.nougat-venv/Scripts/pip install \
    "nougat-ocr==0.1.17" \
    "albumentations<2.0.0" \
    "transformers<VERSION_FROM_STEP_1" \
    "pypdfium2<4.25.0"

echo "Nougat environment ready at .nougat-venv/"
```

(Replace `VERSION_FROM_STEP_1` with the real version you determined. If you're more comfortable with a `.ps1` script given this is a Windows machine, write that instead — whichever you can actually verify works by running it.)

Add `.nougat-venv/` to `.gitignore`.

- [ ] **Step 3: Run the bootstrap and verify the isolated environment actually works**

Run your setup script. Then, using the isolated venv's own interpreter directly (e.g. `.nougat-venv/Scripts/python.exe -c "import nougat; print('ok')"` on Windows), confirm `import nougat` succeeds with NO patches/workarounds needed — this is the entire point of isolation. If it still fails, the pinned versions from Step 1 are wrong; go back and find the actually-correct ones. Do not proceed to Step 4 until a bare `import nougat` succeeds in the isolated environment.

- [ ] **Step 4: Write the standalone extraction script**

`scripts/nougat_extract.py` — runs under the isolated venv, not the main project's. Takes a PDF file path as its one argument, writes Markdown to stdout. Base this on the reference `predict.py` CLI already installed inside `nougat-ocr` (found during this session's earlier investigation, reachable via `.nougat-venv/Scripts/python.exe -c "import predict; print(predict.__file__)"` once the isolated env exists) — that reference implementation's sequence (load checkpoint → `NougatModel.from_pretrained` → `move_to_device` → `model.eval()` → rasterize PDF → build image tensors → `model.inference(...)` → `nougat.postprocessing.markdown_compatible(...)` per page) is the one to follow, adapted to take a file path argument and print joined Markdown to stdout instead of writing `.mmd` files to disk. Verify this script runs standalone (directly with the isolated interpreter against a real PDF) before wiring it into `fulltext.py` — isolate that verification from the subprocess-calling code so failures are easy to attribute to one side or the other.

- [ ] **Step 5: Write the failing test for the subprocess wiring**

```python
def test_extract_nougat_invokes_the_isolated_subprocess(monkeypatch, tmp_path) -> None:
    import researchbridge.benchmark.fulltext as ft

    captured_args = []

    class FakeCompletedProcess:
        stdout = "# Title\n\nSome text with $\\alpha_i$ math."
        returncode = 0

    def fake_run(args, **kwargs):
        captured_args.append(args)
        return FakeCompletedProcess()

    monkeypatch.setattr(ft.subprocess, "run", fake_run)

    result = ft._extract_nougat(b"fake-pdf-bytes")

    assert "alpha_i" in result
    # confirm it invoked the isolated venv's interpreter, not the main one
    assert ".nougat-venv" in captured_args[0][0] or "nougat_extract.py" in " ".join(captured_args[0])


def test_extract_nougat_raises_on_subprocess_failure(monkeypatch) -> None:
    import subprocess

    import researchbridge.benchmark.fulltext as ft

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="model crashed")

    monkeypatch.setattr(ft.subprocess, "run", fake_run)

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        ft._extract_nougat(b"fake-pdf-bytes")
```

Adjust the exact assertions to match your real `_extract_nougat` implementation's argument shape once you write Step 6 — the key behaviors to test are: (a) it calls `subprocess.run` (or equivalent) targeting the isolated venv's interpreter and the extraction script, (b) it returns `_tidy()`-processed stdout on success, (c) it propagates/raises on subprocess failure rather than silently returning empty output (this is the exact failure mode — silent empty output — that one of the six original bugs exhibited; do not repeat it).

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark_fulltext.py -k extract_nougat -v`
Expected: FAIL — `_extract_nougat` still has Task 2's `NotImplementedError` stub

- [ ] **Step 7: Implement `_extract_nougat`**

In `src/researchbridge/benchmark/fulltext.py`, replace Task 2's stub:

```python
import subprocess
import tempfile
from pathlib import Path

NOUGAT_VENV_PYTHON = Path(__file__).resolve().parents[3] / ".nougat-venv" / "Scripts" / "python.exe"
NOUGAT_EXTRACT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "nougat_extract.py"


def _extract_nougat(pdf_bytes: bytes) -> str:
    """Math-aware extraction via Nougat, run in an isolated subprocess.

    nougat-ocr (last released 2023) is incompatible with this project's
    main environment's transformers/albumentations/pypdfium2 versions -
    six independent, unrelated version-drift failures were found
    attempting a direct in-process import (see the design spec's
    Amendment section). This runs Nougat in its own pinned virtual
    environment (.nougat-venv/, set up via scripts/setup_nougat_env.sh)
    as a subprocess instead, so it never touches or fights the main
    project's dependency versions.

    Raises subprocess.CalledProcessError if the isolated extraction
    fails - never silently returns empty output on failure (a real bug
    hit during development: rasterize_paper's own bytes-input bug
    silently produced zero pages rather than erroring).
    """
    if not NOUGAT_VENV_PYTHON.exists():
        raise RuntimeError(
            f"Isolated Nougat environment not found at {NOUGAT_VENV_PYTHON}. "
            "Run scripts/setup_nougat_env.sh first."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(NOUGAT_VENV_PYTHON), str(NOUGAT_EXTRACT_SCRIPT), str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=1800,  # real CPU inference is slow - see the design spec
        )
    finally:
        pdf_path.unlink(missing_ok=True)

    return _tidy(result.stdout)
```

(Adjust the exact `subprocess.run` arguments/error handling to match what Step 5's test actually asserts, and to match Step 4's real script's real argument/output contract.)

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark_fulltext.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add .gitignore scripts/setup_nougat_env.sh scripts/nougat_extract.py src/researchbridge/benchmark/fulltext.py tests/test_benchmark_fulltext.py
git commit -m "feat: implement Nougat extraction via isolated subprocess"
```

Note: `.nougat-venv/` itself is never committed (git-ignored) — only its setup script is.

---

## Task 4 (AMENDED): Live smoke-test the isolated subprocess against a real benchmark PDF, verify `_tidy()` is Markdown-safe

**Files:**
- Modify: `src/researchbridge/benchmark/fulltext.py` (only if Step 3 below finds `_tidy` mangles Markdown)

- [ ] **Step 1: Run a real extraction against a real cached PDF**

Same as originally specified — use `1812.02641` ("Local Conditioning in Undirected Networks"), re-download its PDF (only the PyMuPDF text is cached, not the original PDF bytes):

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

This triggers the model checkpoint's first-run download inside the isolated subprocess — expect this to take a while (the real time budget for CPU inference is still unknown at this point, since no attempt has yet completed end-to-end).

- [ ] **Step 2: Compare against the known-garbled PyMuPDF output**

Check that the math expressions scrambled in `benchmark/fulltext/1812.02641.txt` (the `Φi = eαi / e−αi` matrix, the `∑`/`∏` operators that came out as bare `X`/`Y`) now appear as recognizable LaTeX in the Nougat output. If the output looks wrong (garbled differently, empty, or clearly not math-aware) — or if the subprocess itself fails — STOP and report BLOCKED with full detail (the subprocess's stderr, the exact error). Given the history here, do not attempt more than one self-directed fix before reporting back if something new goes wrong; escalate early rather than repeating the previous pattern.

- [ ] **Step 3: Check `_tidy()` against the real Markdown output**

Same as originally specified: inspect whether `_tidy()`'s blank-line-collapsing regex altered any real Markdown structure incorrectly. Fix only if a real problem is found; re-run Task 3's tests to confirm nothing regressed.

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

**Placeholder scan:** Amended Task 3 still contains a few `VERSION_FROM_STEP_1`-style fill-ins in the setup script and extraction script — same deliberate exception as the original Task 3, now for a different reason: the correct pinned dependency versions and the exact `predict.py`-derived extraction sequence must be discovered from the isolated environment once it exists, not guessed (guessing the original in-process versions is exactly what produced six failures). Steps 1-4 are concrete, runnable discovery/verification actions that produce the information Steps 5-7 need. Every other task has fully concrete code.

**Amendment history:** the original Task 3/4 (direct in-process `nougat`/`torch` import) were attempted, hit six independent dependency-version incompatibilities across three fix rounds with zero successful extractions, and were reverted (commit `1e29af3`, reverting `849fe41`). The amended Tasks 3-4 above (isolated subprocess with a pinned, period-correct environment) replace them. Full history preserved in the spec's "Amendment" section and this plan's git history.

**Type consistency:** `extractor: str = "pymupdf"` and the two-value set `{"pymupdf", "nougat"}` are used identically across `fulltext_path` (Task 1), `extract_text` (Task 2), and `fetch_fulltext` (Task 5). `has_fulltext_nougat`/`fulltext_nougat` naming is consistent between the backend schema (Task 7) and frontend types (Task 8).
