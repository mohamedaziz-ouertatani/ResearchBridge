# Phase 1, Weeks 1-2 — arXiv Connector & Raw Ingestion — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint). This spec covers only the first implementation slice: the arXiv connector, the `NormalizedPaper` representation, and raw ingestion into PostgreSQL with reliability instrumentation, per §8-9 and §45 (weeks 1-2) of the blueprint.

## Scope

In scope:
- `ArxivConnector`: fetches papers from the arXiv API (Atom XML), yields `NormalizedPaper` objects.
- `NormalizedPaper`: canonical in-memory representation, including authors, categories, and other source metadata — even though their dedicated relational tables (`authors`, `paper_categories`, etc.) are deferred to weeks 3-4. This avoids re-deriving the connector's parsing logic later.
- Ingestion pipeline with explicit stages: **fetch → normalize → validate → deduplicate → persist**.
- Reliability: a generic `resume_state` (JSONB, source-defined shape) on each ingestion run, so resumability isn't hardcoded to arXiv's pagination scheme. Structured `ingestion_errors` storage for malformed records (raw payload + error detail), so failures are inspectable, never silent.
- `papers` table (per blueprint §9) and two new operational tables: `ingestion_runs`, `ingestion_errors`.
- Retry/backoff on transient fetch failures (HTTP 429/5xx).
- CLI entrypoint to trigger an ingestion run.
- Tests: connector parsing against recorded fixtures (no live network in the suite), pipeline tests against a real Postgres (docker-compose).

Out of scope (deferred, per blueprint phasing):
- FastAPI / any HTTP server.
- `authors`, `paper_authors`, `paper_categories`, `paper_citations` tables (week 3-4).
- Embeddings / pgvector usage (week 6).
- LLM extraction, `evidence`, `extracted_claims` (week 5+).
- A second data source, Neo4j, or any other relational entity from §14/§40.

## Data model (this slice)

```
papers
------
id                  UUID PRIMARY KEY
source              VARCHAR NOT NULL
source_id           VARCHAR NOT NULL
doi                 VARCHAR NULL
title               TEXT NOT NULL
abstract            TEXT
publication_date    DATE
venue               TEXT
document_type       VARCHAR
language            VARCHAR
url                 TEXT
open_access         BOOLEAN
raw_metadata        JSONB       -- includes authors, categories, and other source-specific fields
ingestion_metadata  JSONB
created_at          TIMESTAMPTZ
updated_at          TIMESTAMPTZ

UNIQUE (source, source_id)
```

```
ingestion_runs
---------------
id                  UUID PRIMARY KEY
source              VARCHAR NOT NULL
status              VARCHAR NOT NULL   -- running | completed | failed
started_at          TIMESTAMPTZ
finished_at         TIMESTAMPTZ NULL
resume_state        JSONB              -- opaque, source-defined (e.g. {"start_index": 200} for arXiv)
records_fetched     INTEGER DEFAULT 0
records_inserted    INTEGER DEFAULT 0
records_duplicate   INTEGER DEFAULT 0
records_failed      INTEGER DEFAULT 0
error_summary       TEXT NULL
```

```
ingestion_errors
------------------
id                  UUID PRIMARY KEY
ingestion_run_id    UUID REFERENCES ingestion_runs(id)
source              VARCHAR NOT NULL
raw_payload         JSONB NOT NULL     -- the offending record, verbatim
error_type          VARCHAR NOT NULL   -- parse_error | validation_error | db_error
error_detail        TEXT NOT NULL
occurred_at         TIMESTAMPTZ
```

`resume_state` is intentionally opaque JSONB at the pipeline/DB level — only the connector interprets its shape, so adding a second source later never requires a schema change here.

## NormalizedPaper

```python
@dataclass
class NormalizedAuthor:
    name: str
    order: int
    orcid: str | None = None

@dataclass
class NormalizedPaper:
    source: str
    source_id: str
    title: str
    abstract: str | None
    publication_date: date | None
    authors: list[NormalizedAuthor]
    categories: list[str]
    doi: str | None = None
    venue: str | None = None
    document_type: str | None = None
    language: str | None = None
    url: str | None = None
    open_access: bool | None = None
    raw_metadata: dict = field(default_factory=dict)
```

Authors/categories live on the object now; the pipeline currently folds them into `papers.raw_metadata` rather than writing to dedicated tables (those tables don't exist yet). This is a one-line change at persist time once week 3-4 lands.

## Pipeline stages

1. **Fetch** — `ArxivConnector.fetch(resume_state) -> (papers: list[NormalizedPaper], next_resume_state, exhausted: bool)`. Wrapped in `tenacity` retry with exponential backoff on transient HTTP errors.
2. **Normalize** — connector's own responsibility; by the time `fetch` returns, results are already `NormalizedPaper`. This stage in the pipeline is a pass-through checkpoint (kept explicit/named per the requested four-stage structure, so future sources that return raw dicts still fit the same pipeline shape).
3. **Validate** — required-field checks (title, source_id non-empty; publication_date parseable). Failures go to `ingestion_errors` with `error_type=validation_error`, not raised.
4. **Deduplicate** — check `(source, source_id)` against what's already persisted in this run + DB; duplicates counted in `records_duplicate`, not treated as errors.
5. **Persist** — insert into `papers`; DB constraint violations (race conditions) caught and logged to `ingestion_errors` with `error_type=db_error`, not fatal to the run.

Each run updates its `ingestion_runs` row incrementally (not just at the end), so a crashed run still leaves an accurate partial record and a usable `resume_state`.

## Testing

- `test_arxiv_connector.py`: parses saved Atom XML fixtures, asserts correct `NormalizedPaper` extraction (including authors/categories), no network calls.
- `test_pipeline.py`: runs the fetch→persist pipeline against a real Postgres (docker-compose), using a fake connector that yields controlled `NormalizedPaper`/malformed inputs, asserting dedup, resume, and error-table behavior.
