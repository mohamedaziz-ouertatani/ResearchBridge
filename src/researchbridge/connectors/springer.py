"""Springer Nature Meta API connector.

Fetches papers from the Springer Nature Meta API (JSON) and normalizes them
into NormalizedPaper objects. Pagination state is expressed as an opaque
resume_state dict of the shape {"start": int} (1-indexed, matching the API's
own `s` parameter) — only this module knows or cares about that shape.

Query syntax note: field-scoped queries like `subject:"Artificial
Intelligence"` are gated behind a premium API tier (verified live - a
free-tier key gets a 403 "premium feature" response). Plain free-text/
phrase queries (e.g. `"machine learning"`) work on the free tier and are
what this connector's default query uses.

Page size note: also verified live - this tier 403s on p=50 and above,
but accepts p<=25. DEFAULT_PAGE_SIZE reflects that free-tier ceiling; a
premium key could raise it.

Date note: `publicationDate` is often a nominal issue/volume cover date
shared identically across many otherwise-unrelated records (e.g. every
article in a December 2026 journal issue reporting "2026-12-01"
regardless of when it actually went online), not a meaningful per-paper
date - verified live, an entire batch came back with one shared value.
`onlineDate` is the real per-record availability date and is what this
connector uses for publication_date, falling back to publicationDate
only when onlineDate is absent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

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
    clean_harvested_abstract,
)

logger = logging.getLogger(__name__)

SPRINGER_META_API_URL = "https://api.springernature.com/meta/v2/json"
DEFAULT_PAGE_SIZE = 25

# No documented per-second rate limit was reachable from this session's docs
# fetch - default to a conservative gap between requests, same politeness
# reasoning as the arXiv connector's throttle.
MIN_REQUEST_INTERVAL_SECONDS = 1.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True  # connection errors, timeouts
        return response.status_code == 429 or response.status_code >= 500
    return False


class SpringerConnector:
    source_name = "springer"

    def __init__(
        self,
        query: str,
        api_key: str | None,
        page_size: int = DEFAULT_PAGE_SIZE,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """query uses the Springer Nature Meta API's free-text query syntax,
        e.g. '"machine learning" OR "artificial intelligence"'. Field-scoped
        queries such as `subject:"..."` require a premium key - see module
        docstring.

        sleep_fn/monotonic_fn are injectable so throttling can be tested
        without real waiting.
        """
        if not api_key:
            raise ValueError("api_key is required (see SPRINGER_META_API_KEY in .env.example)")

        self.query = query
        self.api_key = api_key
        self.page_size = page_size
        self.min_request_interval = min_request_interval
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._last_request_at: float | None = None

    def fetch(self, resume_state: dict[str, Any] | None) -> ConnectorFetchResult:
        start = (resume_state or {}).get("start", 1)

        payload = self._fetch_page(start)
        result_meta = (payload.get("result") or [{}])[0]
        total = int(result_meta.get("total", 0))
        records_displayed = int(result_meta.get("recordsDisplayed", 0))
        records = payload.get("records", [])

        papers = [self._normalize_entry(record) for record in records]

        next_start = start + records_displayed
        exhausted = records_displayed == 0 or next_start > total

        return ConnectorFetchResult(
            papers=papers,
            resume_state={"start": next_start},
            exhausted=exhausted,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_page(self, start: int) -> dict[str, Any]:
        self._throttle()
        response = requests.get(
            SPRINGER_META_API_URL,
            params={
                "q": self.query,
                "api_key": self.api_key,
                "s": start,
                "p": self.page_size,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        """Block until min_request_interval has passed since the last request.

        Applies to retries too (it sits inside the retry-wrapped fetch), so a
        server that is already rate-limiting us doesn't get hammered further.
        """
        if self._last_request_at is not None:
            remaining = self.min_request_interval - (self._monotonic() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def _normalize_entry(self, record: dict[str, Any]) -> NormalizedPaper:
        doi = record.get("doi") or _strip_doi_prefix(record.get("identifier", ""))

        authors = [
            NormalizedAuthor(name=creator.get("creator", ""), order=i, orcid=creator.get("ORCID"))
            for i, creator in enumerate(record.get("creators", []))
        ]

        html_url = None
        for link in record.get("url", []):
            if link.get("format") == "html":
                html_url = link.get("value")
                break

        return NormalizedPaper(
            source=self.source_name,
            source_id=doi,
            title=record.get("title", ""),
            abstract=clean_harvested_abstract(record.get("abstract")),
            publication_date=_parse_date(record.get("onlineDate") or record.get("publicationDate")),
            authors=authors,
            categories=list(record.get("subjects", [])),
            doi=doi,
            venue=record.get("publicationName"),
            document_type=record.get("publicationType"),
            language=record.get("language"),
            url=html_url,
            open_access=_parse_bool(record.get("openaccess")),
            raw_metadata={
                "contentType": record.get("contentType"),
                "keyword": record.get("keyword", []),
                "genre": record.get("genre"),
                "publisher": record.get("publisher"),
                "publisherName": record.get("publisherName"),
                "publicationDate": record.get("publicationDate"),  # nominal cover date, not used as publication_date - see module docstring
            },
        )


def _strip_doi_prefix(identifier: str) -> str:
    """"doi:10.1007/..." -> "10.1007/..."""
    return identifier.removeprefix("doi:")


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
