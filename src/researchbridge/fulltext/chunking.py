"""Splits one PaperFullText section's text into paragraph-level chunks.

fulltext/parse.py's split_sections() preserves line structure within each
section (see its own docstring), and PDF extraction (benchmark/fulltext.py's
_tidy()) collapses runs of blank lines down to exactly one blank line
between paragraphs - so splitting a section's text on blank-line boundaries
recovers real paragraph breaks.

A paragraph under `min_words` words (stray figure/table captions, single
heading fragments that split_sections() didn't recognize as a heading) is
merged into an adjacent paragraph rather than becoming its own chunk, so
answer_question() never surfaces a near-empty "quote".
"""

from __future__ import annotations

import re

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(section_text: str, min_words: int = 10) -> list[str]:
    raw = _BLANK_LINE_RE.split(section_text.strip())
    paragraphs = [p.strip() for p in raw if p.strip()]
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer} {para}".strip() if buffer else para
        if len(candidate.split()) < min_words:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged
