"""Resolves the fetchable PDF URL for one paper, per source (Sec 46).

Only ever called for open_access papers (Task 4's pipeline filters its
selection query on Paper.open_access before calling this) - the
open_access=False check here is a defensive second gate, not the primary
filter.
"""

from __future__ import annotations

from researchbridge.benchmark.fulltext import ARXIV_PDF_URL
from researchbridge.db.models import Paper

# Sources whose `paper.url` is already a genuinely fetchable PDF/full-text
# link (see the design spec's "Per-source PDF URL resolution" table):
# CORE's url is downloadUrl/a fulltext link, Semantic Scholar's is
# openAccessPdf.url (only ever set when open_access is True), Springer's is
# an HTML landing page attempted anyway per the design decision - parsing
# failures are caught and logged downstream, not silently avoided here.
_URL_AS_IS_SOURCES = {"core", "semantic_scholar", "springer"}


def resolve_pdf_url(paper: Paper) -> str | None:
    if not paper.open_access:
        return None
    if paper.source == "arxiv":
        return ARXIV_PDF_URL.format(source_id=paper.source_id)
    if paper.source in _URL_AS_IS_SOURCES:
        return paper.url
    return None
