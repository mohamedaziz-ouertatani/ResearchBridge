"""Canonical in-memory paper representation shared by all source connectors."""

from __future__ import annotations

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


def clean_harvested_abstract(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _MATHML_BLOCK_RE.sub("", raw).strip()
    return cleaned or None


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
