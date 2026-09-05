"""Splits a PDF's extracted text into heuristic sections (Sec 46).

Raw text extraction is delegated entirely to
researchbridge.benchmark.fulltext.extract_text (PyMuPDF-based, already
proven on the assessment-upload path and the benchmark tooling) - this
module adds only heading-based section splitting on top of it, so it
never re-implements or re-tests PDF-to-text extraction itself.

The heading vocabulary below is a starting heuristic (see the design
spec's "Open questions" - it wasn't checked against a large sample of real
PDFs before implementation). Revisit if real corpus runs show a common
heading variant being missed.
"""

from __future__ import annotations

import re

import pymupdf

from researchbridge.benchmark.fulltext import extract_text


class PdfParseError(Exception):
    """Raised when pdf_bytes isn't a parseable PDF, or no extractable text
    was found in it (see extract_text's own pymupdf.FileDataError, and the
    empty-text case checked explicitly below)."""


_SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "approach": "methods",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experimental results": "results",
    "results": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "future work": "conclusion",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "references": "references",
    "bibliography": "references",
}

# Longest phrases first, so "conclusion and future work" matches whole
# before the shorter "conclusion" alias gets a chance to.
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:[0-9]+\.?|[IVXLC]+\.?|[A-Z]\.)?\s*("
    + "|".join(re.escape(k) for k in sorted(_SECTION_ALIASES, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)


def split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    matched_any = False

    for line in lines:
        match = _HEADING_LINE_RE.match(line)
        section_name = _SECTION_ALIASES.get(match.group(1).lower()) if match else None
        # IGNORECASE can match non-ASCII case-fold equivalents (e.g. Turkish
        # dotless i, a PyMuPDF font-encoding artifact) that .lower() doesn't
        # normalize back to the ASCII alias key - .get() treats that rare
        # miss as "not actually a heading" instead of crashing the run.
        if section_name is not None:
            current = section_name
            sections.setdefault(current, [])
            matched_any = True
            continue
        sections[current].append(line)

    if not matched_any:
        return {"body": text.strip()}

    result = {name: "\n".join(section_lines).strip() for name, section_lines in sections.items()}
    return {name: content for name, content in result.items() if content}


def parse_pdf(pdf_bytes: bytes) -> dict[str, str]:
    try:
        text = extract_text(pdf_bytes)
    except pymupdf.FileDataError as exc:
        raise PdfParseError(f"not a valid PDF: {exc}") from exc

    if not text.strip():
        raise PdfParseError("no extractable text found in PDF")

    return split_sections(text)
