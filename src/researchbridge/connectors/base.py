"""Canonical in-memory paper representation shared by all source connectors."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

# Publisher-supplied abstracts (harvested by Semantic Scholar, CORE, and
# fetched directly from Springer) sometimes embed each numeric expression
# twice: a MathML/LaTeX-ish block (e.g. "$$99.88\%\!\pm \!0.22\%$$")
# immediately followed by a plain-text rendering of the same value
# ("99.88 % ± 0.22 %") - a JATS abstract-formatting artifact from the
# originating publisher, not something any of these APIs intend. Verified
# live against real Springer API responses. Strip the markup block and
# keep only the plain-text rendering that follows.
_MATHML_BLOCK_RE = re.compile(r"\$\$.*?\$\$\s*", re.DOTALL)

# CORE-harvested abstracts (source="core") sometimes lose the "&#x"/"&#"
# wrapper off a numeric HTML entity somewhere upstream, leaving the bare
# codepoint digits glued into the text as if they were plain content - e.g.
# a real CORE abstract read "...consumers2019; privacy concerns" and
# "...the gap between 201C;Average Daily Demand201D;..." where the source
# clearly meant "consumers’ privacy" and "the gap between “Average Daily
# Demand”...". Investigated against the real corpus before writing this
# (not guessed): querying every CORE abstract matching a bare 4-digit-plus-
# semicolon pattern showed the overwhelming majority are genuine citation
# years ("Kaiser 2019; Nasim 2019", "Genome Med. 2019;11:70", "Chui et al.
# 2018; Lytras et al. 2018") - a blind "digits+semicolon -> character"
# substitution would corrupt those. Two things distinguish the real
# corruption, both required before this ever substitutes anything:
#   1. the codepoint is a common typographic mark (curly quote, dash,
#      ellipsis) - never a plausible year - restricted to a small verified
#      whitelist, not "any 4-digit number";
#   2. it is glued directly to a word character on at least one side with
#      NO whitespace - every genuine year in the sampled data had a space
#      or punctuation before it ("Oct. 7, 2019;"), while every confirmed
#      corruption instance was glued ("users2019;", "term 201C;Artificial").
# Hex-lettered codepoints (201c/201d/8217/...) that aren't 4 plain decimal
# digits can never collide with a real year at all, so those are repaired
# unconditionally - only the plain-decimal codepoints that coincide with
# plausible years (2018/2019) require the glued-word check.
_STRIPPED_ENTITY_CHARS: dict[str, str] = {
    "2018": "‘", "2019": "’",  # ‘ ’
    "201c": "“", "201d": "”",  # “ ”
    "2013": "–", "2014": "—",  # – —
    "2026": "…",  # …
    "8216": "‘", "8217": "’",
    "8220": "“", "8221": "”",
    "8211": "–", "8212": "—",
    "8230": "…",
}
_UNAMBIGUOUS_CODES = frozenset({"201c", "201d", "8216", "8217", "8220", "8221", "8211", "8212", "8230"})
_STRIPPED_ENTITY_RE = re.compile(
    r"(?P<pre>[A-Za-z])?"
    r"(?P<code>2018|2019|201[Cc]|201[Dd]|2013|2014|2026|8216|8217|8220|8221|8211|8212|8230)"
    r";(?P<post>[A-Za-z])?"
)


def _repair_stripped_html_entities(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        code = match.group("code").lower()
        char = _STRIPPED_ENTITY_CHARS[code]
        pre, post = match.group("pre"), match.group("post")
        if code not in _UNAMBIGUOUS_CODES and not pre and not post:
            # a plain-decimal code (2018/2019) with whitespace on both
            # sides is indistinguishable from a real year - leave it alone
            return match.group(0)
        return f"{pre or ''}{char}{post or ''}"

    return _STRIPPED_ENTITY_RE.sub(repl, text)


def clean_harvested_abstract(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _MATHML_BLOCK_RE.sub("", raw).strip()
    # well-formed entities ("&#x2019;", "&amp;") the source API sent but
    # never decoded - safe, standard-library, touches only real "&...;"
    # sequences
    cleaned = html.unescape(cleaned)
    # entities that already lost their "&#x"/"&#" wrapper before reaching
    # this codebase (see _STRIPPED_ENTITY_RE above)
    cleaned = _repair_stripped_html_entities(cleaned)
    return cleaned.strip() or None


@dataclass
class NormalizedAuthor:
    name: str
    order: int
    orcid: str | None = None


@dataclass
class NormalizedPaper:
    """Canonical representation a connector must produce, regardless of source.

    Authors and categories live here even though their dedicated relational
    tables (paper_authors, paper_categories) are deferred to a later phase —
    connectors shouldn't need to be rewritten when those tables land.
    """

    source: str
    source_id: str
    title: str
    abstract: str | None
    publication_date: date | None
    authors: list[NormalizedAuthor] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    doi: str | None = None
    venue: str | None = None
    document_type: str | None = None
    language: str | None = None
    url: str | None = None
    open_access: bool | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class FetchResult(Protocol):
    papers: list[NormalizedPaper]
    resume_state: dict[str, Any] | None
    exhausted: bool


@dataclass
class ConnectorFetchResult:
    papers: list[NormalizedPaper]
    resume_state: dict[str, Any] | None
    exhausted: bool


class Connector(Protocol):
    """A source connector: knows how to turn one provider's API into NormalizedPaper objects."""

    source_name: str

    def fetch(self, resume_state: dict[str, Any] | None) -> ConnectorFetchResult:
        """Fetch the next batch of papers.

        resume_state is opaque to everything except this connector — the
        pipeline and database only ever pass it back unchanged. This keeps
        pagination/cursor semantics source-specific without leaking into
        shared schema or pipeline code.
        """
        ...
