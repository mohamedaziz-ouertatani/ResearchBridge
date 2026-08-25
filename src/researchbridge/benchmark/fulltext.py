"""Fetches and caches full text for the benchmark papers only.

The corpus stores abstracts (Sec 8), which is enough for embeddings but not
for annotation: Sec 25 asks for explicitly-stated limitations, remaining
gaps, and supporting passages with sections, and those live in a paper's
Discussion and Future Work - not in a 240-word abstract.

This is the narrow, justified case Sec 46 allows for PDF processing: 40
benchmark papers, fetched once and cached to disk. It is deliberately NOT
wired into the ingestion pipeline.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from collections import namedtuple
from pathlib import Path

import requests

from researchbridge.connectors.arxiv import MIN_REQUEST_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{source_id}"
REQUEST_TIMEOUT = 60


_EXTRACTOR_FILENAMES = {
    "pymupdf": "{source_id}.txt",
    "nougat": "{source_id}.nougat.md",
}


def fulltext_path(output_dir: Path, source_id: str, extractor: str = "pymupdf") -> Path:
    if extractor not in _EXTRACTOR_FILENAMES:
        raise ValueError(f"extractor must be one of {sorted(_EXTRACTOR_FILENAMES)}, got {extractor!r}")
    return output_dir / _EXTRACTOR_FILENAMES[extractor].format(source_id=source_id)


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
    if extractor == "nougat":
        # Nougat's OCR discards figures entirely - they only come back via
        # this second, independent PyMuPDF pass over the same PDF bytes,
        # spliced in at the page markers nougat_extract.py leaves behind.
        images_by_page = save_images(extract_images(response.content), output_dir, source_id)
        text = splice_images(text, images_by_page, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


PageImage = namedtuple("PageImage", ["page", "index", "ext", "data"])


def extract_images(pdf_bytes: bytes) -> list[PageImage]:
    """Pull embedded raster images out of a PDF, page by page (0-indexed).

    Imported lazily, same as _extract_pymupdf: the rest of the benchmark
    tooling shouldn't pay for PyMuPDF unless this step actually runs.
    """
    import pymupdf

    images: list[PageImage] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            for index, img in enumerate(page.get_images(full=True)):
                base = doc.extract_image(img[0])
                images.append(PageImage(page=page.number, index=index, ext=base["ext"], data=base["image"]))
    return images


def image_dir(output_dir: Path, source_id: str) -> Path:
    return output_dir.parent / "images" / source_id


def save_images(images: list[PageImage], output_dir: Path, source_id: str) -> dict[int, list[str]]:
    """Write each image to disk; return {page: [filenames]} for splice_images."""
    directory = image_dir(output_dir, source_id)
    by_page: dict[int, list[str]] = {}
    for image in images:
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"p{image.page:04d}_{image.index}.{image.ext}"
        (directory / filename).write_bytes(image.data)
        by_page.setdefault(image.page, []).append(filename)
    return by_page


_PAGE_MARKER_RE = re.compile(r"<!--PAGE:(\d+)-->\n*")


def splice_images(markdown: str, images_by_page: dict[int, list[str]], source_id: str) -> str:
    """Replace each page marker with that page's figures, or drop it if there are none.

    Nougat keeps no anchor finer than "which page a figure was on" - so
    "this figure was on page N" is the honest placement to make, not a guess
    at which paragraph it sat beside.
    """

    def replace(match: re.Match[str]) -> str:
        page = int(match.group(1))
        filenames = images_by_page.get(page, [])
        if not filenames:
            return ""
        figures = "\n\n".join(
            f"![figure, page {page + 1}](/api/benchmark/papers/{source_id}/images/{name})"
            for name in filenames
        )
        return f"\n\n{figures}\n\n"

    return _PAGE_MARKER_RE.sub(replace, markdown)


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


def _extract_pymupdf(pdf_bytes: bytes) -> str:
    """Extract readable text from a PDF, page by page.

    Imported lazily so the rest of the benchmark tooling doesn't pay for
    PyMuPDF, which is only needed for this one step.
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]

    return _tidy("\n\n".join(pages))


NOUGAT_VENV_PYTHON = Path(__file__).resolve().parents[3] / ".nougat-venv" / "Scripts" / "python.exe"
NOUGAT_EXTRACT_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "nougat_extract.py"


def _extract_nougat(pdf_bytes: bytes) -> str:
    """Math-aware extraction via Nougat, run in an isolated subprocess.

    nougat-ocr (last released 2023) is incompatible with this project's
    main environment's transformers/albumentations/pypdfium2 versions -
    six independent, unrelated version-drift failures were found
    attempting a direct in-process import (see the design spec's
    Amendment section). This runs Nougat in its own pinned virtual
    environment (.nougat-venv/, set up via scripts/setup_nougat_env.ps1)
    as a subprocess instead, so it never touches or fights the main
    project's dependency versions.

    Raises RuntimeError (wrapping the subprocess's captured stderr) if the
    isolated extraction fails - never silently returns empty output on
    failure (a real bug hit during development: rasterize_paper's own
    bytes-input bug silently produced zero pages rather than erroring).
    """
    if not NOUGAT_VENV_PYTHON.exists():
        raise RuntimeError(
            f"Isolated Nougat environment not found at {NOUGAT_VENV_PYTHON}. "
            "Run scripts/setup_nougat_env.ps1 first."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        pdf_path = Path(f.name)

    try:
        result = subprocess.run(
            [str(NOUGAT_VENV_PYTHON), str(NOUGAT_EXTRACT_SCRIPT), str(pdf_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            # 7200s (2h): the small-model/CPU-era 1800s value killed a clean
            # 36-page paper mid-extraction on the current GPU/base-model
            # setup - the 79-page paper used to calibrate this took 63
            # minutes (3779s) by itself, so 1800s was never enough even for
            # a mid-sized paper once the model changed. See design spec.
            timeout=7200,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Nougat extraction failed (exit {exc.returncode}): {(exc.stderr or '')[-1000:]}"
        ) from exc
    finally:
        pdf_path.unlink(missing_ok=True)

    return normalize_nougat_markdown(_tidy(result.stdout))


_TABLE_FLOAT_RE = re.compile(r"\\(?:begin|end)\{table\*?\}(?:\[[^\]]*\])?")
# \hline \hline (a heavier top/bottom rule, LaTeX's cheap stand-in for
# \toprule/\bottomrule) is common - 23 of 40 benchmark papers use it - but
# mathpix-markdown-it's tabular parser rejects it outright and abandons the
# whole table to raw text, confirmed by rendering real output through the
# actual parser. \hline is decorative only, so collapsing repeats to one
# changes nothing about the table's content.
_DOUBLED_HLINE_RE = re.compile(r"\\hline(?:\s+\\hline)+")
_TABULAR_OPEN = "\\begin{tabular}"
_TABULAR_CLOSE = "\\end{tabular}"


def normalize_nougat_markdown(text: str) -> str:
    """Repair Nougat's structural OCR defects, leaving its content untouched.

    Nougat writes Mathpix Markdown (.mmd), a documented format that renderers
    parse natively. What they cannot be expected to handle is output that is
    not valid MMD in the first place: a ``` fence that never closes (which
    turns the whole rest of the paper into one code block), a \\begin{table}
    whose partner tag was dropped, a table that simply stops mid-row, or a
    table that landed inside a code fence.

    Only structure is touched - no character of prose, math or table content
    is added, removed or rewritten - so the repaired text still says exactly
    what Nougat read off the page.
    """
    lines = text.split("\n")

    # A fence that never closes hides everything after it. Pair them off; if
    # one is left open at the end, that is the one to drop.
    unpaired = -1
    for index, line in enumerate(lines):
        if line.startswith("```"):
            unpaired = index if unpaired == -1 else -1
    if unpaired != -1:
        del lines[unpaired]

    out = _TABLE_FLOAT_RE.sub("", "\n".join(lines))
    out = _DOUBLED_HLINE_RE.sub(r"\\hline", out)

    # A table whose \end{tabular} was dropped runs on and swallows the prose
    # after it. It ends where the OCR moved on: at the next blank line.
    if out.count(_TABULAR_OPEN) > out.count(_TABULAR_CLOSE):
        deficit = out.count(_TABULAR_OPEN) - out.count(_TABULAR_CLOSE)
        depth = 0
        repaired: list[str] = []
        for line in out.split("\n"):
            depth += line.count(_TABULAR_OPEN) - line.count(_TABULAR_CLOSE)
            if depth > 0 and deficit > 0 and not line.strip():
                repaired.append(_TABULAR_CLOSE)
                depth -= 1
                deficit -= 1
            repaired.append(line)
        out = "\n".join(repaired)
        if deficit > 0:
            out += "\n" + f"{_TABULAR_CLOSE}\n" * deficit

    return _unfence_pseudocode(_lift_tabulars_out_of_fences(out))


# An algorithm listing carries inline math on nearly every step; a real code
# listing carries none. Three spans is comfortably above what a stray \(...\)
# in a comment would produce, and well below the dozens a real algorithm has.
_PSEUDOCODE_MATH_SPANS = 3


def _unfence_pseudocode(text: str) -> str:
    """Unwrap algorithm listings so their math can render.

    Nougat fences algorithm pseudocode as a code block, but its steps are
    written in inline math. Nothing renders inside a fence, so the reader is
    shown raw \\(\\mathcal{U}_{0}\\)... instead of typeset symbols. Genuine
    code listings have no math and are left fenced, where they belong.

    Steps are joined with markdown hard breaks (two trailing spaces) so one
    step stays on one line instead of collapsing into a running paragraph.
    """
    parts = text.split("```")
    # Odd indices are fenced blocks; even indices are the prose between them.
    # An unpaired fence is already dropped upstream, so the split is even.
    rebuilt: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            rebuilt.append(part)
        elif part.count("\\(") >= _PSEUDOCODE_MATH_SPANS:
            steps = [line.rstrip() for line in part.strip("\n").split("\n")]
            rebuilt.append("\n\n" + "  \n".join(steps) + "\n\n")
        else:
            # Genuine listing: put its fences back exactly as they were.
            rebuilt.append(f"```{part}```")

    return "".join(rebuilt)


def _lift_tabulars_out_of_fences(text: str) -> str:
    """Close the code fence around a table that was OCR'd inside one.

    Nougat's fences land in the wrong place often enough that a real table
    ends up inside a listing, where every renderer shows it as literal LaTeX
    rather than a table. The fence is the mistake, not the table.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            index += 1
            continue
        if not in_fence or _TABULAR_OPEN not in line:
            out.append(line)
            index += 1
            continue

        out.append("```")
        depth = 0
        while index < len(lines):
            depth += lines[index].count(_TABULAR_OPEN) - lines[index].count(_TABULAR_CLOSE)
            out.append(lines[index])
            index += 1
            if depth <= 0:
                break
        out.append("```")

    return "\n".join(out)


# Math-heavy PDFs often set equations in fonts with custom glyph-to-code
# mappings (e.g. a symbol font where code point 0x11 is drawn as an
# operator glyph). A text extractor has no way to recover the intended
# Unicode for those, so it emits raw control characters or Private Use
# Area code points instead - not a rendering bug, the correct character
# genuinely isn't recoverable this way. Stripping them turns garbled runs
# like "x(tensor)k<PUA>" into "x(tensor)k": the surrounding prose reads
# cleanly, and the lost symbol becomes a gap rather than a broken glyph.
# Recovering the equation itself would need math-aware OCR (e.g. Mathpix,
# Nougat), well beyond what plain text extraction can do.
_UNRENDERABLE_RANGES = [
    (0x00, 0x08),  # C0 controls, excluding \t (0x09) \n (0x0A)
    (0x0B, 0x0C),
    (0x0E, 0x1F),
    (0x7F, 0x9F),  # DEL and C1 controls
    (0xE000, 0xF8FF),  # Private Use Area
    (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
    (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
    (0xFFFD, 0xFFFD),  # replacement character
]
_UNRENDERABLE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _UNRENDERABLE_RANGES) + "]")


def _tidy(text: str) -> str:
    """Collapse the ragged whitespace PDF extraction leaves behind, and drop
    characters a text extractor cannot have gotten right (see _UNRENDERABLE_RANGES).

    Line structure is kept (annotators read this, and section headings
    depend on line breaks); only runs of blank lines and trailing spaces
    on each line are normalized.
    """
    text = _UNRENDERABLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def throttle() -> None:
    """Space PDF downloads the same way the API connector spaces its requests."""
    time.sleep(MIN_REQUEST_INTERVAL_SECONDS)
