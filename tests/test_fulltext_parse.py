from __future__ import annotations

import pytest

from researchbridge.fulltext.parse import PdfParseError, parse_pdf, split_sections


def test_split_sections_splits_on_known_headings() -> None:
    text = (
        "Introduction\n"
        "This paper studies X.\n"
        "Methods\n"
        "We use approach Y.\n"
        "Limitations\n"
        "This only works offline.\n"
    )
    sections = split_sections(text)
    assert sections["introduction"] == "This paper studies X."
    assert sections["methods"] == "We use approach Y."
    assert sections["limitations"] == "This only works offline."


def test_split_sections_recognizes_common_heading_variants() -> None:
    text = "Methodology\nDetails here.\nConclusion and Future Work\nWe conclude.\n"
    sections = split_sections(text)
    assert sections["methods"] == "Details here."
    assert sections["conclusion"] == "We conclude."


def test_split_sections_handles_numbered_headings() -> None:
    text = "1. Introduction\nSome text.\n2. Related Work\nOther text.\n"
    sections = split_sections(text)
    assert sections["introduction"] == "Some text."
    assert sections["related_work"] == "Other text."


def test_split_sections_falls_back_to_body_when_no_heading_matches() -> None:
    text = "Just some prose with no recognizable heading lines at all."
    sections = split_sections(text)
    assert sections == {"body": text}


def test_split_sections_drops_empty_sections() -> None:
    text = "Introduction\n\nMethods\nReal content.\n"
    sections = split_sections(text)
    assert "introduction" not in sections
    assert sections["methods"] == "Real content."


def test_split_sections_ignores_a_heading_line_that_ignorecase_matches_but_lower_cannot_normalize() -> None:
    # Turkish dotless-i (U+0131): re.IGNORECASE matches it against ASCII
    # "introduction" during matching, but match.group(1).lower() does not
    # fold it back to ASCII "i" - a real PyMuPDF extraction artifact that
    # crashed a production run with KeyError: 'introductıon' before
    # split_sections() switched to a dict .get() fallback.
    text = "Introductıon\nThis paper studies X.\n"
    sections = split_sections(text)
    assert "introduction" not in sections
    assert sections["body"] == text.strip()


def test_parse_pdf_wraps_extract_text_and_splits_it(monkeypatch) -> None:
    import researchbridge.fulltext.parse as parse_module

    monkeypatch.setattr(parse_module, "extract_text", lambda pdf_bytes: "Introduction\nHello.\n")

    sections = parse_pdf(b"fake-pdf-bytes")
    assert sections == {"introduction": "Hello."}


def test_parse_pdf_raises_pdfparseerror_on_invalid_pdf_bytes() -> None:
    with pytest.raises(PdfParseError):
        parse_pdf(b"not a real pdf")


def test_parse_pdf_raises_pdfparseerror_on_empty_extracted_text(monkeypatch) -> None:
    import researchbridge.fulltext.parse as parse_module

    monkeypatch.setattr(parse_module, "extract_text", lambda pdf_bytes: "   ")

    with pytest.raises(PdfParseError):
        parse_pdf(b"fake-pdf-bytes")
