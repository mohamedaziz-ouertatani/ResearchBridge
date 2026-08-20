"""arXiv API connector.

Fetches papers from the arXiv API (Atom XML) and normalizes them into
NormalizedPaper objects. Pagination state is expressed as an opaque
resume_state dict of the shape {"start_index": int} — only this module
knows or cares about that shape.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import feedparser
import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from researchbridge.connectors.base import (
    ConnectorFetchResult,
    NormalizedAuthor,
    NormalizedPaper,
)

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE = 100


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True  # connection errors, timeouts
        return response.status_code == 429 or response.status_code >= 500
    return False


class ArxivConnector:
    source_name = "arxiv"

    def __init__(self, search_query: str, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        """search_query uses arXiv's query syntax, e.g. "cat:cs.LG" or
        "cat:cs.LG OR cat:cs.AI"."""
        self.search_query = search_query
        self.page_size = page_size

    def fetch(self, resume_state: dict[str, Any] | None) -> ConnectorFetchResult:
        start_index = (resume_state or {}).get("start_index", 0)

        feed_text = self._fetch_page(start_index)
        parsed = feedparser.parse(feed_text)

        total_results = int(parsed.feed.get("opensearch_totalresults", 0))
        entries = parsed.entries

        papers = [self._normalize_entry(entry) for entry in entries]

        next_start = start_index + len(entries)
        exhausted = len(entries) == 0 or next_start >= total_results

        return ConnectorFetchResult(
            papers=papers,
            resume_state={"start_index": next_start},
            exhausted=exhausted,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_page(self, start_index: int) -> bytes:
        response = requests.get(
            ARXIV_API_URL,
            params={
                "search_query": self.search_query,
                "start": start_index,
                "max_results": self.page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.content

    def _normalize_entry(self, entry: Any) -> NormalizedPaper:
        source_id = _extract_arxiv_id(entry.get("id", ""))

        authors = [
            NormalizedAuthor(name=author.get("name", ""), order=i)
            for i, author in enumerate(entry.get("authors", []))
        ]

        categories = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]
        primary_category = entry.get("arxiv_primary_category", {}).get("term")

        publication_date = _parse_date(entry.get("published"))

        pdf_url = None
        abs_url = entry.get("link")
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")

        return NormalizedPaper(
            source=self.source_name,
            source_id=source_id,
            title=_clean_whitespace(entry.get("title", "")),
            abstract=_clean_whitespace(entry.get("summary", "")) or None,
            publication_date=publication_date,
            authors=authors,
            categories=categories,
            doi=entry.get("arxiv_doi"),
            venue=entry.get("arxiv_journal_ref"),
            document_type="preprint",
            language="en",
            url=abs_url,
            open_access=True,
            raw_metadata={
                "primary_category": primary_category,
                "categories": categories,
                "comment": entry.get("arxiv_comment"),
                "pdf_url": pdf_url,
                "updated": entry.get("updated"),
                "authors": [{"name": a.name} for a in authors],
            },
        )


def _extract_arxiv_id(entry_id: str) -> str:
    """"http://arxiv.org/abs/2101.00001v2" -> "2101.00001" (version stripped)."""
    match = re.search(r"abs/([^v]+)", entry_id)
    base = match.group(1) if match else entry_id
    return base.rstrip("/")


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(value: str | None) -> Any:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None
