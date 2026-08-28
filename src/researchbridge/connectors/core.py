"""CORE API v3 connector (https://api.core.ac.uk/v3/search/works).

CORE only indexes open-access works, so open_access is always True for
every normalized paper - there is no per-record flag to read, unlike
Springer/Semantic Scholar.

Auth note: CORE v3 requires a free API key on every request, sent as
`Authorization: Bearer <key>` - a different style from Springer's
query-param key, but the same "required, not optional" shape (see
api_key validation below). Register at https://core.ac.uk/services/api.

Docs verification note: unlike the Springer/Semantic Scholar connectors,
this one could NOT be "verified live" against the real API during
development - api.core.ac.uk/docs/v3 and core.ac.uk/documentation/api
both returned HTTP 403 to automated fetches. The request/response shape
below (offset/limit pagination, `results`/`totalHits` envelope, Bearer
auth, per-record fields) is reconstructed from secondary sources
(existing open-source CORE API clients) rather than the primary docs.
Treat field names as best-effort until a real ingestion run against a
live key confirms them, and revisit this docstring once that happens.

Rate limit note: CORE's free tier is documented informally as roughly
5 single requests per 10 seconds. MIN_REQUEST_INTERVAL_SECONDS defaults
to a conservative 2.0s gap between requests - same reasoning as the
other connectors' throttles, adjust if a live run proves it too
aggressive or unnecessarily slow.
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
)

logger = logging.getLogger(__name__)

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
DEFAULT_PAGE_SIZE = 100

MIN_REQUEST_INTERVAL_SECONDS = 2.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True  # connection errors, timeouts
        return response.status_code == 429 or response.status_code >= 500
    return False


class CoreConnector:
    source_name = "core"

    def __init__(
        self,
        query: str,
        api_key: str | None,
        page_size: int = DEFAULT_PAGE_SIZE,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """query uses the CORE API's free-text search syntax, e.g.
        'machine learning AND artificial intelligence'.

        sleep_fn/monotonic_fn are injectable so throttling can be tested
        without real waiting.
        """
        if not api_key:
            raise ValueError("api_key is required (see CORE_API_KEY in .env.example)")

        self.query = query
        self.api_key = api_key
        self.page_size = page_size
        self.min_request_interval = min_request_interval
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._last_request_at: float | None = None

    def fetch(self, resume_state: dict[str, Any] | None) -> ConnectorFetchResult:
        offset = (resume_state or {}).get("offset", 0)

        payload = self._fetch_page(offset)
        total_hits = int(payload.get("totalHits", 0))
        results = payload.get("results", [])

        papers = [self._normalize_entry(record) for record in results]

        next_offset = offset + len(results)
        exhausted = not results or next_offset >= total_hits

        return ConnectorFetchResult(
            papers=papers,
            resume_state={"offset": next_offset},
            exhausted=exhausted,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_page(self, offset: int) -> dict[str, Any]:
        self._throttle()
        response = requests.get(
            CORE_SEARCH_URL,
            params={"q": self.query, "limit": self.page_size, "offset": offset},
            headers={"Authorization": f"Bearer {self.api_key}"},
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
        authors = [
            NormalizedAuthor(name=author.get("name", ""), order=i, orcid=None)
            for i, author in enumerate(record.get("authors") or [])
        ]

        year_published = record.get("yearPublished")
        publication_date = _parse_datetime(record.get("publishedDate"))
        if publication_date is None and year_published:
            publication_date = date(int(year_published), 1, 1)

        url = record.get("downloadUrl")
        if not url:
            fulltext_urls = record.get("sourceFulltextUrls") or []
            url = fulltext_urls[0] if fulltext_urls else None

        language = record.get("language")
        language_code = language.get("code") if isinstance(language, dict) else language

        core_id = record.get("id")

        return NormalizedPaper(
            source=self.source_name,
            source_id=str(core_id),
            title=record.get("title", ""),
            abstract=record.get("abstract") or None,
            publication_date=publication_date,
            authors=authors,
            categories=list(record.get("subjects") or []),
            doi=record.get("doi") or None,
            venue=record.get("publisher") or None,
            document_type=record.get("documentType"),
            language=language_code,
            url=url,
            open_access=True,  # CORE indexes only open-access works - no per-record flag exists
            raw_metadata={
                "yearPublished": year_published,
                "sourceFulltextUrls": record.get("sourceFulltextUrls"),
            },
        )


def _parse_datetime(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None
