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
import time
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


def fetch_fulltext(source_id: str, output_dir: Path, session: requests.Session | None = None) -> str:
    """Download one paper's PDF and cache its extracted text. Returns the text."""
    path = fulltext_path(output_dir, source_id)
    if path.exists():
        return path.read_text(encoding="utf-8")

    http = session or requests
    response = http.get(
        ARXIV_PDF_URL.format(source_id=source_id),
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "ResearchBridge/0.1 (benchmark annotation; solo research project)"},
    )
    response.raise_for_status()

    text = extract_text(response.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def extract_text(pdf_bytes: bytes) -> str:
    """Extract readable text from a PDF, page by page.

    Imported lazily so the rest of the benchmark tooling doesn't pay for
    PyMuPDF, which is only needed for this one step.
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]

    return _tidy("\n\n".join(pages))


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
