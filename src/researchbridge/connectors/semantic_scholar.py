"""Semantic Scholar Graph API connector (bulk search endpoint).

Uses /graph/v1/paper/search/bulk rather than the regular /paper/search:
the regular endpoint caps total pagination at 10,000 results and paginates
by numeric offset, while bulk search paginates via an opaque `token`
cursor with no such cap - a better fit for exhaustive ingestion, the same
role the Springer connector's `{"start": int}` resume_state plays there.

No API key is required - this connector runs unauthenticated by default,
same as arXiv. SEMANTIC_SCHOLAR_API_KEY is optional: when present it's
sent as the `x-api-key` header, which raises the caller's rate-limit
tier without changing response shape or normalization. Per explicit
instruction, the request throttle stays conservative (matching the
unauthenticated shared-pool rate) regardless of whether a key is
supplied, even though Semantic Scholar currently publishes a faster
introductory rate (1 req/sec) for authenticated keys - not assumed
reliable enough to lean on by default.

Verified live against the real API (unauthenticated, query='"machine
learning"'): bulk search returns exactly 1000 records per page as
documented, and consecutive token-based pages don't overlap. Two real
data-completeness gaps, both handled gracefully rather than being API
bugs:
- `publicationDate` is null for ~14% of results (Semantic Scholar only
  has year-level precision for those records) - publication_date ends
  up None for them, same as any paper missing the field.
- `fieldsOfStudy` is null for ~42% of results (incomplete classification
  coverage on Semantic Scholar's side) - categories ends up an empty
  list for them, same handling as any paper with no categories.

A later real ingestion pass (11+ consecutive pages, unauthenticated) hit a
429 after the connector's retry budget was exhausted, at the throttle
value that first testing considered safe - see MIN_REQUEST_INTERVAL_SECONDS.
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

SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = "title,abstract,year,publicationDate,authors,externalIds,venue,publicationTypes,openAccessPdf,fieldsOfStudy"

# Verified live 2026-08-22: 3.0s (the nominal ~100 req/5min average) still
# got a sustained unauthenticated run 429'd after 11 consecutive requests -
# the shared pool doesn't tolerate a steady run at its average rate, only
# bursts under it. Doubled to 6.0s. Kept as the default even with an API
# key supplied - see module docstring.
MIN_REQUEST_INTERVAL_SECONDS = 6.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True  # connection errors, timeouts
        return response.status_code == 429 or response.status_code >= 500
    return False


class SemanticScholarConnector:
    source_name = "semantic_scholar"

    def __init__(
        self,
        query: str,
        api_key: str | None = None,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """query uses the Semantic Scholar bulk search endpoint's boolean text
        query syntax, e.g. '"machine learning" | "artificial intelligence"'.

        api_key is optional (see SEMANTIC_SCHOLAR_API_KEY in .env.example) -
        this connector runs unauthenticated by default, same as arXiv.

        sleep_fn/monotonic_fn are injectable so throttling can be tested
        without real waiting.
        """
        self.query = query
        self.api_key = api_key
        self.min_request_interval = min_request_interval
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._last_request_at: float | None = None

    def fetch(self, resume_state: dict[str, Any] | None) -> ConnectorFetchResult:
        token = (resume_state or {}).get("token")

        payload = self._fetch_page(token)
        records = payload.get("data", [])
        next_token = payload.get("token")

        papers = [self._normalize_entry(record) for record in records]

        exhausted = not records or next_token is None

        return ConnectorFetchResult(
            papers=papers,
            resume_state={"token": next_token},
            exhausted=exhausted,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_page(self, token: str | None) -> dict[str, Any]:
        self._throttle()
        params: dict[str, Any] = {"query": self.query, "fields": FIELDS}
        if token is not None:
            params["token"] = token
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = requests.get(
            SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
            params=params,
            headers=headers,
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
        external_ids = record.get("externalIds") or {}
        doi = external_ids.get("DOI")

        authors = [
            NormalizedAuthor(name=author.get("name", ""), order=i, orcid=None)
            for i, author in enumerate(record.get("authors", []))
        ]

        publication_types = record.get("publicationTypes") or []
        document_type = publication_types[0] if publication_types else None

        open_access_pdf = record.get("openAccessPdf")
        paper_id = record.get("paperId", "")
        # Most records have no open-access PDF, but every record has a
        # Semantic Scholar landing page - fall back to that so "read online"
        # still has somewhere to send the reader.
        url = open_access_pdf.get("url") if open_access_pdf else None
        if not url and paper_id:
            url = f"https://www.semanticscholar.org/paper/{paper_id}"

        return NormalizedPaper(
            source=self.source_name,
            source_id=paper_id,
            title=record.get("title", ""),
            abstract=record.get("abstract") or None,
            publication_date=_parse_date(record.get("publicationDate")),
            authors=authors,
            categories=list(record.get("fieldsOfStudy") or []),
            doi=doi,
            venue=record.get("venue") or None,
            document_type=document_type,
            language=None,  # not provided by the bulk search endpoint's default fields
            url=url,
            open_access=open_access_pdf is not None,
            raw_metadata={
                "externalIds": external_ids,
                "year": record.get("year"),
                "publicationTypes": publication_types,
            },
        )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
