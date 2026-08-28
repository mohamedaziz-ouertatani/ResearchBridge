"""Fetches and persists intra-corpus citation edges from Semantic Scholar.

Only Semantic Scholar exposes citation data among the connectors this
project has (arXiv, Springer, CORE have no citations endpoint). Uses the
per-paper details endpoint (not the bulk search endpoint the ingestion
connector uses), since citation lists aren't part of bulk search's
default fields.

An edge is only ever persisted when BOTH the citing and cited paper
already exist in the corpus (matched by source="semantic_scholar",
source_id=<S2 paperId>) - most Semantic Scholar citations/references
will point outside this corpus, and that's expected, not a bug.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from researchbridge.db.models import Paper, PaperCitation

SEMANTIC_SCHOLAR_PAPER_DETAILS_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
CITATION_FIELDS = "citations.paperId,references.paperId"

# Same shared-pool caution as connectors/semantic_scholar.py's throttle -
# see that module's docstring for the live-testing history behind 6.0s.
MIN_REQUEST_INTERVAL_SECONDS = 6.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True  # connection errors, timeouts
        return response.status_code == 429 or response.status_code >= 500
    return False


@dataclass
class RawCitationsPayload:
    citing_source_ids: list[str] = field(default_factory=list)
    """S2 paperIds of papers that cite this paper (from the "citations" field)."""
    cited_source_ids: list[str] = field(default_factory=list)
    """S2 paperIds of papers this paper cites (from the "references" field)."""


@dataclass
class CitationResolution:
    seed_paper_id: uuid.UUID
    edges: list[tuple[uuid.UUID, uuid.UUID]]
    """(citing_paper_id, cited_paper_id) pairs, both already resolved to internal Paper ids."""
    unresolved_citing: int
    """Citation source_ids with no matching in-corpus paper."""
    unresolved_cited: int
    """Reference source_ids with no matching in-corpus paper."""


class SemanticScholarCitationFetcher:
    def __init__(
        self,
        api_key: str | None = None,
        min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api_key = api_key
        self.min_request_interval = min_request_interval
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn
        self._last_request_at: float | None = None

    def fetch_raw(self, source_id: str) -> RawCitationsPayload:
        payload = self._fetch_page(source_id)
        citing = [c["paperId"] for c in (payload.get("citations") or []) if c.get("paperId")]
        cited = [r["paperId"] for r in (payload.get("references") or []) if r.get("paperId")]
        return RawCitationsPayload(citing_source_ids=citing, cited_source_ids=cited)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _fetch_page(self, source_id: str) -> dict[str, Any]:
        self._throttle()
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = requests.get(
            SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id=source_id),
            params={"fields": CITATION_FIELDS},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            remaining = self.min_request_interval - (self._monotonic() - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()


def resolve_citations(session: Session, paper: Paper, payload: RawCitationsPayload) -> CitationResolution:
    """Pure read: resolves citing/cited S2 paperIds against existing Paper rows.

    Does not write anything - see persist_citation_edges for the write side.
    Kept separate so a dry run can show what WOULD be created.
    """
    all_ids = set(payload.citing_source_ids) | set(payload.cited_source_ids)
    if not all_ids:
        return CitationResolution(seed_paper_id=paper.id, edges=[], unresolved_citing=0, unresolved_cited=0)

    rows = session.execute(
        select(Paper.source_id, Paper.id).where(Paper.source == "semantic_scholar", Paper.source_id.in_(all_ids))
    ).all()
    id_by_source_id = {source_id: internal_id for source_id, internal_id in rows}

    edges: list[tuple[uuid.UUID, uuid.UUID]] = []
    unresolved_citing = 0
    for source_id in payload.citing_source_ids:
        internal_id = id_by_source_id.get(source_id)
        if internal_id is None:
            unresolved_citing += 1
            continue
        edges.append((internal_id, paper.id))

    unresolved_cited = 0
    for source_id in payload.cited_source_ids:
        internal_id = id_by_source_id.get(source_id)
        if internal_id is None:
            unresolved_cited += 1
            continue
        edges.append((paper.id, internal_id))

    return CitationResolution(
        seed_paper_id=paper.id, edges=edges, unresolved_citing=unresolved_citing, unresolved_cited=unresolved_cited
    )


def persist_citation_edges(session: Session, resolution: CitationResolution) -> int:
    """Inserts a PaperCitation row per resolved edge, skipping ones that already
    exist (the unique constraint on citing/cited/source already enforces this -
    same try/skip-on-IntegrityError pattern as ingestion/pipeline.py's _persist).

    Does not commit - the caller controls transaction boundaries.
    """
    created = 0
    for citing_id, cited_id in resolution.edges:
        try:
            with session.begin_nested():
                session.add(
                    PaperCitation(
                        citing_paper_id=citing_id,
                        cited_paper_id=cited_id,
                        source="semantic_scholar",
                        confidence="high",
                    )
                )
        except IntegrityError:
            continue
        created += 1
    return created
