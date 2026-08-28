# Citation Graph Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the currently-empty `paper_citations` table via a new Semantic Scholar citation-fetching pipeline, and add a per-paper citation graph view on the paper detail page.

**Architecture:** A new `citations` package fetches per-paper citation/reference lists from the Semantic Scholar Graph API's paper-details endpoint, resolves each citing/cited paperId against papers already in the corpus, and persists only intra-corpus edges into the existing `paper_citations` table. A new `GET /api/papers/{id}/citations` endpoint reads that table directly (no computation, unlike the similarity graph) and a new `CitationGraph` frontend component renders it with the same `react-force-graph` machinery `SimilarityGraph` already uses, direction (cites/cited-by) taking the place of distance-based coloring.

**Tech Stack:** Python (SQLAlchemy, requests, tenacity), FastAPI/Pydantic, Next.js/TypeScript, react-force-graph.

**Spec:** `docs/superpowers/specs/2026-08-28-citation-graph-exploration-design.md`

## Global Constraints

- Semantic Scholar only as the citation source. No CrossRef, no other connector.
- Intra-corpus edges only: an edge is persisted only when both `citing_paper_id` and `cited_paper_id` already exist as rows in `papers` (matched by `source="semantic_scholar", source_id=<S2 paperId>`). Citations to papers outside the corpus are silently skipped, never stubbed.
- No schema/migration changes. `paper_citations` keeps its existing shape (`citing_paper_id`, `cited_paper_id` both `NOT NULL` FKs, unique on `(citing_paper_id, cited_paper_id, source)`).
- `confidence="high"` on every inserted `PaperCitation` row — source-asserted, not inferred.
- No admin-panel trigger, no `*_runs` tracking table for citation fetching. CLI-only (`rb-citations-fetch`), dry-run by default, `--save` to persist — same tier as `rb-gaps-detect`.
- Same throttle/retry discipline as `connectors/semantic_scholar.py`: 6.0s minimum request interval, retry on 429/5xx via `tenacity`, each module keeps its own small `_is_retryable` (existing per-connector convention — not shared/imported across modules).
- No frontend automated tests (no test infra exists yet). Frontend work is manually verified in-browser.

---

### Task 1: Citation fetch HTTP layer

**Files:**
- Create: `src/researchbridge/citations/__init__.py`
- Create: `src/researchbridge/citations/fetch.py`
- Test: `tests/test_citation_fetch.py`

**Interfaces:**
- Produces: `RawCitationsPayload` dataclass (`citing_source_ids: list[str]`, `cited_source_ids: list[str]`), `SemanticScholarCitationFetcher` class with `fetch_raw(source_id: str) -> RawCitationsPayload`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_citation_fetch.py`:

```python
from researchbridge.citations.fetch import (
    SEMANTIC_SCHOLAR_PAPER_DETAILS_URL,
    RawCitationsPayload,
    SemanticScholarCitationFetcher,
)

import responses


@responses.activate
def test_fetch_raw_returns_citing_and_cited_source_ids() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(
        responses.GET,
        url,
        json={
            "paperId": "abc123",
            "citations": [{"paperId": "cit1"}, {"paperId": "cit2"}],
            "references": [{"paperId": "ref1"}],
        },
        status=200,
    )

    fetcher = SemanticScholarCitationFetcher()
    result = fetcher.fetch_raw("abc123")

    assert result == RawCitationsPayload(citing_source_ids=["cit1", "cit2"], cited_source_ids=["ref1"])


@responses.activate
def test_fetch_raw_handles_missing_citations_and_references() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(responses.GET, url, json={"paperId": "abc123"}, status=200)

    fetcher = SemanticScholarCitationFetcher()
    result = fetcher.fetch_raw("abc123")

    assert result == RawCitationsPayload(citing_source_ids=[], cited_source_ids=[])


@responses.activate
def test_sends_api_key_header_when_provided() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(responses.GET, url, json={"paperId": "abc123"}, status=200)

    fetcher = SemanticScholarCitationFetcher(api_key="fake-key")
    fetcher.fetch_raw("abc123")

    assert responses.calls[0].request.headers["x-api-key"] == "fake-key"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_citation_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.citations'`

- [ ] **Step 3: Write minimal implementation**

Create `src/researchbridge/citations/__init__.py` (empty file).

Create `src/researchbridge/citations/fetch.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_citation_fetch.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/citations/__init__.py src/researchbridge/citations/fetch.py tests/test_citation_fetch.py
git commit -m "feat: fetch citation edges from the Semantic Scholar per-paper API"
```

---

### Task 2: Resolve and persist against a real corpus

**Files:**
- Modify: `tests/test_citation_fetch.py`

**Interfaces:**
- Consumes: `RawCitationsPayload`, `CitationResolution`, `resolve_citations`, `persist_citation_edges` from Task 1.
- Consumes (test-only): `session_factory`/`engine` fixtures from `tests/conftest.py`; `Paper`, `PaperCitation` from `researchbridge.db.models`.

This task adds DB-backed tests for the resolve/persist functions written in Task 1 (they need real `Paper` rows to resolve against, which `responses`-mocked HTTP tests can't provide) - no new production code, just test coverage proving the intra-corpus resolution and dedup behavior work end to end.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_citation_fetch.py`:

```python
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from researchbridge.citations.fetch import CitationResolution, persist_citation_edges, resolve_citations
from researchbridge.db.models import Paper, PaperCitation


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def _add_s2_paper(session, source_id: str, title: str = "A Paper") -> Paper:
    paper = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id=source_id, title=title)
    session.add(paper)
    session.commit()
    return paper


def test_resolve_citations_matches_in_corpus_papers_only(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")
    cited = _add_s2_paper(session, "cited1")

    payload = RawCitationsPayload(citing_source_ids=["citing1", "not-in-corpus"], cited_source_ids=["cited1"])
    resolution = resolve_citations(session, target, payload)

    assert set(resolution.edges) == {(citing.id, target.id), (target.id, cited.id)}
    assert resolution.unresolved_citing == 1
    assert resolution.unresolved_cited == 0


def test_resolve_citations_with_no_ids_returns_empty_resolution(session) -> None:
    target = _add_s2_paper(session, "target1")

    resolution = resolve_citations(session, target, RawCitationsPayload())

    assert resolution == CitationResolution(seed_paper_id=target.id, edges=[], unresolved_citing=0, unresolved_cited=0)


def test_persist_citation_edges_creates_rows(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")

    resolution = CitationResolution(seed_paper_id=target.id, edges=[(citing.id, target.id)], unresolved_citing=0, unresolved_cited=0)
    created = persist_citation_edges(session, resolution)
    session.commit()

    assert created == 1
    row = session.query(PaperCitation).one()
    assert (row.citing_paper_id, row.cited_paper_id, row.source, row.confidence) == (citing.id, target.id, "semantic_scholar", "high")


def test_persist_citation_edges_skips_edges_that_already_exist(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")
    session.add(PaperCitation(citing_paper_id=citing.id, cited_paper_id=target.id, source="semantic_scholar", confidence="high"))
    session.commit()

    resolution = CitationResolution(seed_paper_id=target.id, edges=[(citing.id, target.id)], unresolved_citing=0, unresolved_cited=0)
    created = persist_citation_edges(session, resolution)
    session.commit()

    assert created == 0
    assert session.query(PaperCitation).count() == 1
```

Add `RawCitationsPayload` to the existing import line from `researchbridge.citations.fetch` at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_citation_fetch.py -v`
Expected: If Postgres is running at `TEST_DATABASE_URL` (`docker compose up -d`), these should actually PASS immediately since Task 1's implementation already supports this - this step is a sanity check that the new tests exercise real behavior, not a red step. If Postgres isn't reachable, the tests will be SKIPPED with "Postgres not reachable" - start it and rerun.

- [ ] **Step 3: Write minimal implementation**

None needed - Task 1's `resolve_citations`/`persist_citation_edges` already satisfy these tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_citation_fetch.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add tests/test_citation_fetch.py
git commit -m "test: cover citation resolution/persistence against a real corpus"
```

---

### Task 3: Batch runner over the corpus

**Files:**
- Create: `src/researchbridge/citations/batch.py`
- Test: `tests/test_citation_batch.py`

**Interfaces:**
- Consumes: `SemanticScholarCitationFetcher.fetch_raw`, `resolve_citations`, `persist_citation_edges` from Task 1.
- Produces: `BatchSummary` dataclass, `run_all(session, fetcher, force=False, save=False) -> BatchSummary`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_citation_batch.py`:

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.citations.batch import BatchSummary, run_all
from researchbridge.citations.fetch import RawCitationsPayload
from researchbridge.db.models import Paper, PaperCitation


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


@dataclass
class FakeFetcher:
    """Returns a fixed payload per source_id, recording every call made."""

    payloads: dict[str, RawCitationsPayload] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def fetch_raw(self, source_id: str) -> RawCitationsPayload:
        self.calls.append(source_id)
        return self.payloads.get(source_id, RawCitationsPayload())


def _add_s2_paper(session, source_id: str) -> Paper:
    paper = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id=source_id, title=f"Paper {source_id}")
    session.add(paper)
    session.commit()
    return paper


def test_run_all_fetches_and_saves_edges_for_every_semantic_scholar_paper(session) -> None:
    a = _add_s2_paper(session, "a")
    b = _add_s2_paper(session, "b")
    fetcher = FakeFetcher(payloads={"a": RawCitationsPayload(cited_source_ids=["b"])})

    summary = run_all(session, fetcher, save=True)

    assert sorted(fetcher.calls) == ["a", "b"]
    assert summary == BatchSummary(papers_seen=2, papers_failed=0, edges_created=1, edges_already_existed=0)
    assert session.query(PaperCitation).count() == 1


def test_run_all_dry_run_does_not_persist(session) -> None:
    _add_s2_paper(session, "a")
    _add_s2_paper(session, "b")
    fetcher = FakeFetcher(payloads={"a": RawCitationsPayload(cited_source_ids=["b"])})

    summary = run_all(session, fetcher, save=False)

    assert summary.edges_created == 1  # what WOULD be created
    assert session.query(PaperCitation).count() == 0  # nothing actually persisted


def test_run_all_ignores_non_semantic_scholar_papers(session) -> None:
    session.add(Paper(id=uuid.uuid4(), source="arxiv", source_id="x", title="Not S2"))
    session.commit()
    fetcher = FakeFetcher()

    summary = run_all(session, fetcher, save=True)

    assert fetcher.calls == []
    assert summary.papers_seen == 0


def test_run_all_skips_papers_with_existing_outgoing_edges_unless_forced(session) -> None:
    a = _add_s2_paper(session, "a")
    b = _add_s2_paper(session, "b")
    session.add(PaperCitation(citing_paper_id=a.id, cited_paper_id=b.id, source="semantic_scholar", confidence="high"))
    session.commit()
    fetcher = FakeFetcher()

    summary = run_all(session, fetcher, save=True)
    assert sorted(fetcher.calls) == ["b"]  # a already has an outgoing edge, skipped

    fetcher2 = FakeFetcher()
    run_all(session, fetcher2, save=True, force=True)
    assert sorted(fetcher2.calls) == ["a", "b"]  # force reprocesses everything


def test_run_all_continues_past_a_failing_paper(session) -> None:
    a = _add_s2_paper(session, "a")
    _add_s2_paper(session, "b")

    class RaisingFetcher:
        def fetch_raw(self, source_id: str) -> RawCitationsPayload:
            if source_id == a.source_id:
                raise RuntimeError("boom")
            return RawCitationsPayload()

    summary = run_all(session, RaisingFetcher(), save=True)

    assert summary.papers_seen == 2
    assert summary.papers_failed == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_citation_batch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.citations.batch'`

- [ ] **Step 3: Write minimal implementation**

Create `src/researchbridge/citations/batch.py`:

```python
"""Batch citation fetching over every Semantic Scholar paper in the corpus.

Idempotent by default: a paper with at least one outgoing PaperCitation edge
is skipped on the next run (--force reprocesses it) - same
"no new tracking table, reuse what already answers the question" principle
gaps/batch.py already applies to candidate_gaps.seed_paper_id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from researchbridge.citations.fetch import persist_citation_edges, resolve_citations
from researchbridge.db.models import Paper, PaperCitation

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    papers_seen: int = 0
    papers_failed: int = 0
    edges_created: int = 0
    edges_already_existed: int = 0


def run_all(session: Session, fetcher, force: bool = False, save: bool = False) -> BatchSummary:
    summary = BatchSummary()

    for paper in _select_target_papers(session, force):
        summary.papers_seen += 1
        try:
            payload = fetcher.fetch_raw(paper.source_id)
            resolution = resolve_citations(session, paper, payload)
        except Exception:
            logger.exception("Citation fetch failed for paper %s", paper.id)
            summary.papers_failed += 1
            continue

        attempted = len(resolution.edges)
        if save:
            created = persist_citation_edges(session, resolution)
            session.commit()
        else:
            created = attempted  # dry run: report what WOULD be created

        summary.edges_created += created
        summary.edges_already_existed += attempted - created

    return summary


def _select_target_papers(session: Session, force: bool) -> list[Paper]:
    query = select(Paper).where(Paper.source == "semantic_scholar")
    if not force:
        already_done = exists().where(
            PaperCitation.citing_paper_id == Paper.id, PaperCitation.source == "semantic_scholar"
        )
        query = query.where(~already_done)
    return list(session.execute(query).scalars())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_citation_batch.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/citations/batch.py tests/test_citation_batch.py
git commit -m "feat: batch-run citation fetching over the corpus"
```

---

### Task 4: CLI entry point

**Files:**
- Create: `src/researchbridge/citations/cli_fetch.py`
- Modify: `pyproject.toml`
- Test: `tests/test_citations_cli.py`

**Interfaces:**
- Consumes: `SemanticScholarCitationFetcher`, `resolve_citations`, `persist_citation_edges` (Task 1), `run_all` (Task 3).
- Produces: `build_parser()`, `validate_args(args, parser=None)`, `main()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_citations_cli.py` (mirrors `tests/test_gaps_cli_detect.py`):

```python
from __future__ import annotations

import pytest

from researchbridge.citations.cli_fetch import build_parser, validate_args


def test_accepts_a_source_id_alone() -> None:
    args = build_parser().parse_args(["abc123"])
    validate_args(args)  # must not raise


def test_accepts_all_alone() -> None:
    args = build_parser().parse_args(["--all"])
    validate_args(args)  # must not raise


def test_rejects_source_id_and_all_together() -> None:
    args = build_parser().parse_args(["abc123", "--all"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_neither_source_id_nor_all() -> None:
    args = build_parser().parse_args([])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_force_without_all() -> None:
    args = build_parser().parse_args(["abc123", "--force"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_accepts_force_with_all() -> None:
    args = build_parser().parse_args(["--all", "--force"])
    validate_args(args)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_citations_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.citations.cli_fetch'`

- [ ] **Step 3: Write minimal implementation**

Create `src/researchbridge/citations/cli_fetch.py`:

```python
"""Fetches citation edges for one Semantic Scholar paper, or every such paper
with --all. Dry run by default: prints what it would create. --save persists
PaperCitation rows.
"""

from __future__ import annotations

import argparse
import logging
import os

from sqlalchemy import select

from researchbridge.citations.batch import run_all
from researchbridge.citations.fetch import SemanticScholarCitationFetcher, persist_citation_edges, resolve_citations
from researchbridge.config import load_config
from researchbridge.db.models import Paper
from researchbridge.db.session import make_engine, make_session_factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch citation edges from Semantic Scholar.")
    parser.add_argument("source_id", type=str, nargs="?", default=None, help="Semantic Scholar paperId of the seed paper (omit with --all)")
    parser.add_argument("--all", action="store_true", help="run over every Semantic Scholar paper without existing outgoing edges")
    parser.add_argument("--force", action="store_true", help="with --all, reprocess papers that already have outgoing edges")
    parser.add_argument("--save", action="store_true", help="persist PaperCitation rows (default: dry run)")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    parser = parser or build_parser()
    if args.all and args.source_id:
        parser.error("pass either a source_id or --all, not both")
    if not args.all and not args.source_id:
        parser.error("source_id is required unless --all is given")
    if args.force and not args.all:
        parser.error("--force only applies with --all")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    session = make_session_factory(make_engine())()
    try:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        fetcher = SemanticScholarCitationFetcher(api_key=api_key)

        if args.all:
            _run_batch(session, fetcher, args)
        else:
            _run_single(session, fetcher, args)
    finally:
        session.close()


def _run_single(session, fetcher, args: argparse.Namespace) -> None:
    paper = session.execute(
        select(Paper).where(Paper.source == "semantic_scholar", Paper.source_id == args.source_id)
    ).scalar_one_or_none()
    if paper is None:
        print(f"No Semantic Scholar paper with id {args.source_id} in the corpus.")
        return

    payload = fetcher.fetch_raw(args.source_id)
    resolution = resolve_citations(session, paper, payload)

    print(f"Seed: {paper.title}")
    print(f"Resolved edges: {len(resolution.edges)}")
    print(f"Unresolved citing (outside corpus): {resolution.unresolved_citing}")
    print(f"Unresolved cited (outside corpus): {resolution.unresolved_cited}")

    if args.save:
        created = persist_citation_edges(session, resolution)
        session.commit()
        print(f"Saved {created} new citation edge(s).")


def _run_batch(session, fetcher, args: argparse.Namespace) -> None:
    mode = "saving" if args.save else "dry run"
    print(f"Running citation fetch over every Semantic Scholar paper ({mode})...")

    summary = run_all(session, fetcher, force=args.force, save=args.save)

    print(f"Papers seen:            {summary.papers_seen}")
    print(f"Papers failed:          {summary.papers_failed}")
    print(f"Edges created:          {summary.edges_created}")
    print(f"Edges already existed:  {summary.edges_already_existed}")


if __name__ == "__main__":
    main()
```

Add to `pyproject.toml`'s `[project.scripts]` section, alongside `rb-gaps-detect`:

```toml
rb-citations-fetch = "researchbridge.citations.cli_fetch:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_citations_cli.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/citations/cli_fetch.py pyproject.toml tests/test_citations_cli.py
git commit -m "feat: add rb-citations-fetch CLI"
```

---

### Task 5: Citation graph API endpoint

**Files:**
- Modify: `src/researchbridge/api/schemas.py`
- Modify: `src/researchbridge/api/routes.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `CitationNodeOut`, `CitationEdgeOut`, `CitationGraphOut` (Pydantic schemas); `GET /api/papers/{paper_id}/citations` route returning `CitationGraphOut`.
- Consumes: `PaperCitation`, `Paper` from `researchbridge.db.models`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (after the existing `test_similar_papers_*` tests):

```python
from researchbridge.db.models import PaperCitation


def test_paper_citations_returns_cites_and_cited_by_edges(client, session) -> None:
    center = _add_paper(session, "center", title="Center Paper")
    citing = _add_paper(session, "citing", title="Citing Paper")
    cited = _add_paper(session, "cited", title="Cited Paper")
    session.add(PaperCitation(citing_paper_id=citing.id, cited_paper_id=center.id, source="semantic_scholar", confidence="high"))
    session.add(PaperCitation(citing_paper_id=center.id, cited_paper_id=cited.id, source="semantic_scholar", confidence="high"))
    session.commit()

    body = client.get(f"/api/papers/{center.id}/citations").json()

    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {str(center.id), str(citing.id), str(cited.id)}
    center_node = next(n for n in body["nodes"] if n["id"] == str(center.id))
    assert center_node["type"] == "center"

    edges = {(e["source"], e["target"], e["direction"]) for e in body["edges"]}
    assert edges == {
        (str(citing.id), str(center.id), "cited_by"),
        (str(center.id), str(cited.id), "cites"),
    }


def test_paper_citations_empty_when_no_edges(client, session) -> None:
    paper = _add_paper(session, "lonely")

    body = client.get(f"/api/papers/{paper.id}/citations").json()

    assert body["nodes"] == [{"id": str(paper.id), "type": "center", "title": paper.title}]
    assert body["edges"] == []


def test_paper_citations_404s_for_unknown_paper(client) -> None:
    assert client.get(f"/api/papers/{uuid.uuid4()}/citations").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -k paper_citations -v`
Expected: FAIL with 404 (route doesn't exist yet) on all three tests

- [ ] **Step 3: Write minimal implementation**

In `src/researchbridge/api/schemas.py`, add (near `SimilarityGraphOut`):

```python
class CitationNodeOut(BaseModel):
    id: str
    type: Literal["center", "paper"]
    title: str


class CitationEdgeOut(BaseModel):
    source: str
    target: str
    direction: Literal["cites", "cited_by"]


class CitationGraphOut(BaseModel):
    nodes: list[CitationNodeOut]
    edges: list[CitationEdgeOut]
```

In `src/researchbridge/api/routes.py`:

1. Update the model import line:

```python
from researchbridge.db.models import Author, Embedding, ExtractedClaim, Paper, PaperAuthor, PaperCategory, PaperCitation
```

2. Update the schema import line to add `CitationGraphOut`:

```python
from researchbridge.api.schemas import CitationGraphOut, CorpusStats, ExtractedClaimOut, PaperPage, PaperSummary, SearchHit
```

3. Add the route (after `similar_papers`):

```python
@router.get("/papers/{paper_id}/citations", response_model=CitationGraphOut)
def paper_citations(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> CitationGraphOut:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")
    return _build_citation_graph(session, paper)


def _build_citation_graph(session: Session, paper: Paper) -> CitationGraphOut:
    from researchbridge.api.schemas import CitationEdgeOut, CitationNodeOut

    cited_by_rows = session.execute(
        select(PaperCitation.citing_paper_id, Paper.title)
        .join(Paper, Paper.id == PaperCitation.citing_paper_id)
        .where(PaperCitation.cited_paper_id == paper.id)
    ).all()
    cites_rows = session.execute(
        select(PaperCitation.cited_paper_id, Paper.title)
        .join(Paper, Paper.id == PaperCitation.cited_paper_id)
        .where(PaperCitation.citing_paper_id == paper.id)
    ).all()

    nodes = [CitationNodeOut(id=str(paper.id), type="center", title=paper.title)]
    edges = []
    for citing_id, title in cited_by_rows:
        nodes.append(CitationNodeOut(id=str(citing_id), type="paper", title=title))
        edges.append(CitationEdgeOut(source=str(citing_id), target=str(paper.id), direction="cited_by"))
    for cited_id, title in cites_rows:
        nodes.append(CitationNodeOut(id=str(cited_id), type="paper", title=title))
        edges.append(CitationEdgeOut(source=str(paper.id), target=str(cited_id), direction="cites"))

    return CitationGraphOut(nodes=nodes, edges=edges)
```

(The inline `from researchbridge.api.schemas import CitationEdgeOut, CitationNodeOut` inside `_build_citation_graph` avoids a second top-of-file import line churn; move both up to the top-level schemas import alongside `CitationGraphOut` if that reads better at implementation time - purely a style call, not a behavior difference.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -k paper_citations -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest -q`
Expected: all tests pass (no regressions in existing `/papers/*` route tests)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/routes.py tests/test_api.py
git commit -m "feat: add GET /api/papers/{id}/citations endpoint"
```

---

### Task 6: Extract the shared ForceGraph2D loader

**Files:**
- Create: `frontend/lib/forceGraphLoader.ts`
- Modify: `frontend/components/SimilarityGraph.tsx`

**Interfaces:**
- Produces: `loadForceGraph2D(): Promise<ComponentType<...>>` (moved as-is from `SimilarityGraph.tsx`).
- Consumes (Task 8): `CitationGraph.tsx` will import this same function.

No behavior change - this is a pure extraction so Task 8 doesn't duplicate the THREE/AFRAME-stubbing dance. No new tests (no frontend test infra); verified by the existing SimilarityGraph still rendering correctly in Task 9's manual check.

- [ ] **Step 1: Create the shared loader module**

Create `frontend/lib/forceGraphLoader.ts` by moving these exact declarations out of `frontend/components/SimilarityGraph.tsx` (the `loadForceGraph2D` function and its `ForceGraphInstance` type, unchanged):

```typescript
import type { ComponentType } from "react";

// react-force-graph draws to a <canvas> and reads `window` at import time -
// it cannot run during server-side rendering.
//
// The "react-force-graph" barrel unconditionally imports its VR/AR bundles
// alongside ForceGraph2D, and those assume the A-Frame ecosystem's usual
// script-tag setup: window.AFRAME and window.THREE already populated before
// they load. Without that they throw ReferenceErrors at module-evaluation
// time and crash the whole page, even though this app only ever renders the
// 2D graph and never touches the VR/AR code paths. Populate window.THREE
// with the real "three" module (a direct dependency in package.json, pinned
// to the version react-force-graph itself resolves, so an upgrade of either
// package can't silently break this import) so any eager
// `class X extends THREE.Y` evaluates correctly, and stub window.AFRAME
// with a no-op proxy since nothing here ever registers a real A-Frame scene.
export function loadForceGraph2D() {
  const globalWindow = typeof window !== "undefined" ? (window as unknown as { AFRAME?: unknown; THREE?: unknown }) : undefined;
  if (globalWindow && !globalWindow.AFRAME) {
    globalWindow.AFRAME = new Proxy({}, { get: () => () => undefined });
  }
  const threeReady =
    globalWindow && !globalWindow.THREE
      ? // "three" ships no type declarations here and this import exists purely to
        // populate a runtime global, not to use typed exports - see comment above.
        // @ts-expect-error -- untyped module, see comment above
        import("three").then((THREE) => {
          // ES module namespace objects are frozen; several three.js loader
          // add-ons patch themselves onto window.THREE directly
          // (`THREE.ColladaLoader = ...`), so it must be a plain, extensible object.
          globalWindow.THREE = { ...THREE };
        })
      : Promise.resolve();
  return threeReady.then(() => import("react-force-graph")).then((mod) => mod.ForceGraph2D);
}

// next/dynamic() strips the generic type parameters off ForceGraph2D, so its
// accessor props would otherwise widen to `{}`. Each consumer recasts to its
// own concrete node/link shape - see SimilarityGraph.tsx/CitationGraph.tsx.
export type ForceGraphInstance = {
  d3Force: (name: "link" | "charge" | "center" | string) =>
    | { strength?: (v: number) => unknown; distance?: (v: number | ((link: never) => number)) => unknown }
    | undefined;
};
```

- [ ] **Step 2: Update SimilarityGraph.tsx to import from the shared module**

In `frontend/components/SimilarityGraph.tsx`, delete the `loadForceGraph2D` function definition and the `ForceGraphInstance` type it previously defined locally (lines 26-53 in the current file), and add this import near the top instead:

```typescript
import { loadForceGraph2D, type ForceGraphInstance } from "@/lib/forceGraphLoader";
```

The local `ForceGraph2D = dynamic(loadForceGraph2D, { ssr: false }) as unknown as ComponentType<{...}>` declaration and its concrete prop typing (nodeId, nodeLabel, nodeColor, etc.) stay in `SimilarityGraph.tsx` unchanged - only the loader function and its instance type move out.

- [ ] **Step 3: Manually verify no regression**

Run: `npx tsc --noEmit` from `frontend/` (per this project's convention of typechecking frontend changes)
Expected: no new errors

Start the dev server and open an existing assessment's detail page; confirm the similarity graph still renders and node click/side-panel behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/forceGraphLoader.ts frontend/components/SimilarityGraph.tsx
git commit -m "refactor: extract shared ForceGraph2D loader for reuse by CitationGraph"
```

---

### Task 7: Frontend API client for the citations endpoint

**Files:**
- Modify: `frontend/lib/api.ts`

**Interfaces:**
- Produces: `CitationNode`, `CitationEdge`, `CitationGraphData` types; `api.citations(id: string): Promise<CitationGraphData>`.

- [ ] **Step 1: Add types and the API call**

In `frontend/lib/api.ts`, add near the other type definitions (after `SearchHit`):

```typescript
export type CitationNode = {
  id: string;
  type: "center" | "paper";
  title: string;
};

export type CitationEdge = {
  source: string;
  target: string;
  direction: "cites" | "cited_by";
};

export type CitationGraphData = {
  nodes: CitationNode[];
  edges: CitationEdge[];
};
```

Add to the `api` object, alongside `similar`:

```typescript
  citations: (id: string) => get<CitationGraphData>(`/api/papers/${id}/citations`),
```

- [ ] **Step 2: Verify with a typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors (this is a pure addition, nothing else references it yet)

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat: add citations() to the frontend API client"
```

---

### Task 8: CitationGraph component

**Files:**
- Create: `frontend/components/CitationGraph.tsx`

**Interfaces:**
- Consumes: `loadForceGraph2D`, `ForceGraphInstance` (Task 6); `api.citations`, `CitationGraphData`, `CitationNode`, `CitationEdge` (Task 7).

- [ ] **Step 1: Write the component**

Create `frontend/components/CitationGraph.tsx`:

```typescript
"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type CitationEdge, type CitationGraphData, type CitationNode } from "@/lib/api";
import { loadForceGraph2D, type ForceGraphInstance } from "@/lib/forceGraphLoader";

type GraphLink = { source: string; target: string; direction: CitationEdge["direction"] };

const ForceGraph2D = dynamic(loadForceGraph2D, {
  ssr: false,
}) as unknown as ComponentType<{
  graphData: { nodes: CitationNode[]; links: GraphLink[] };
  nodeId: string;
  nodeLabel: (node: CitationNode) => string | HTMLElement;
  nodeColor: (node: CitationNode) => string;
  nodeVal: (node: CitationNode) => number;
  linkColor: (link: GraphLink) => string;
  linkDirectionalArrowLength: number;
  linkDirectionalArrowRelPos: number;
  onNodeClick: (node: CitationNode) => void;
  onBackgroundClick: () => void;
  width: number;
  height: number;
  ref?: (instance: ForceGraphInstance | null) => void;
}>;

// Same spacing convention as SimilarityGraph - see that component's comment
// for why these values (not the library defaults) were chosen.
const CHARGE_STRENGTH = -200;
const LINK_DISTANCE = 120;

function configureForces(instance: ForceGraphInstance | null) {
  if (!instance) return;
  instance.d3Force("charge")?.strength?.(CHARGE_STRENGTH);
  instance.d3Force("link")?.distance?.(LINK_DISTANCE);
}

// Same XSS-avoidance reasoning as SimilarityGraph's nodeTooltipLabel: paper
// titles are untrusted (arXiv/Springer/Semantic Scholar/CORE/user uploads),
// and float-tooltip calls .html() on a plain string, so build the tooltip
// as a real element with textContent instead of returning raw text.
function nodeTooltipLabel(node: CitationNode): string | HTMLElement {
  if (typeof document === "undefined") return node.title;
  const el = document.createElement("span");
  el.textContent = node.title;
  return el;
}

const GRAPH_HEIGHT = 420;

export function CitationGraph({ paperId }: { paperId: string }) {
  const [data, setData] = useState<CitationGraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CitationNode | null>(null);
  const [graphWidth, setGraphWidth] = useState(0);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    setSelected(null);
    api
      .citations(paperId)
      .then(setData)
      .catch(() => setError("The citation graph isn't available."));
  }, [paperId]);

  const measuredContainerRef = useCallback((el: HTMLDivElement | null) => {
    resizeObserverRef.current?.disconnect();
    resizeObserverRef.current = null;
    if (!el) return;
    setGraphWidth(el.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setGraphWidth(width);
    });
    observer.observe(el);
    resizeObserverRef.current = observer;
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.edges.map((e) => ({ source: e.source, target: e.target, direction: e.direction })),
    };
  }, [data]);

  if (error) {
    return (
      <section className="border-t border-[var(--rule)] py-8">
        <span className="eyebrow">citation graph</span>
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">{error}</p>
      </section>
    );
  }

  if (!data) {
    return <p className="eyebrow py-6">loading citations…</p>;
  }

  const hasEdges = data.edges.length > 0;

  return (
    <section className="border-t border-[var(--rule)] py-8">
      <span className="eyebrow">citation graph</span>

      {!hasEdges ? (
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">
          No citation data yet for this paper.
        </p>
      ) : (
        <div className="mt-4 flex flex-col gap-4 sm:flex-row">
          <div ref={measuredContainerRef} className="h-[420px] flex-1 overflow-hidden border border-[var(--rule)]">
            {graphWidth > 0 && (
              <ForceGraph2D
                ref={configureForces}
                graphData={graphData}
                nodeId="id"
                nodeLabel={nodeTooltipLabel}
                nodeColor={(node: CitationNode) => (node.type === "center" ? "var(--ink)" : "var(--ink-soft)")}
                nodeVal={(node: CitationNode) => (node.type === "center" ? 8 : 4)}
                linkColor={(link: GraphLink) => (link.direction === "cites" ? "var(--near)" : "var(--far)")}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                onNodeClick={(node: CitationNode) => setSelected(node.type === "paper" ? node : null)}
                onBackgroundClick={() => setSelected(null)}
                width={graphWidth}
                height={GRAPH_HEIGHT}
              />
            )}
          </div>

          {selected && (
            <aside className="w-full shrink-0 border border-[var(--rule)] bg-[var(--panel)] p-4 sm:w-64">
              <Link
                href={`/papers/${selected.id}`}
                className="text-[0.9375rem] leading-[1.5] underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
              >
                {selected.title}
              </Link>
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors. This component isn't wired into any page yet (Task 9), so this step only confirms it compiles standalone.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/CitationGraph.tsx
git commit -m "feat: add CitationGraph component"
```

---

### Task 9: Wire CitationGraph into the paper detail page

**Files:**
- Modify: `frontend/app/papers/[id]/page.tsx`

**Interfaces:**
- Consumes: `CitationGraph` (Task 8).

- [ ] **Step 1: Add the import and render the section**

In `frontend/app/papers/[id]/page.tsx`, add the import:

```typescript
import { CitationGraph } from "@/components/CitationGraph";
```

Add the section after the existing `<section className="mt-16">...nearest in meaning...</section>` block, still inside the `<article>`:

```tsx
          <CitationGraph paperId={id} />
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors

- [ ] **Step 3: Manually verify in the browser**

Start the dev server (`npm run dev` from `frontend/`). Navigate to a paper detail page:
- A paper with no `paper_citations` rows shows "No citation data yet for this paper."
- After manually inserting a `PaperCitation` row for a test paper (e.g. via `psql` or a quick Python shell against the dev DB) and reloading, the graph renders with the center node and satellite node(s), colored by direction; clicking a satellite node opens the side panel with a working link to that paper's detail page.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/papers/[id]/page.tsx
git commit -m "feat: show the citation graph on the paper detail page"
```
