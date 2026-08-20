# ResearchBridge

Research intelligence / research-to-impact decision support platform. See
`ResearchBridge.md` for the full architectural blueprint — that document is
the source of truth for all design decisions.

This repo currently implements Phase 1, weeks 1-2 only: the arXiv connector,
the `NormalizedPaper` representation, and a reliable raw-ingestion pipeline
into PostgreSQL. See `docs/superpowers/specs/2026-08-20-phase1-arxiv-ingestion-design.md`
for the design of this slice.

## Setup

```bash
uv sync
docker compose up -d
cp .env.example .env
uv run alembic upgrade head
```

## Run ingestion

```bash
uv run rb-ingest --search-query "cat:cs.LG OR cat:cs.AI" --page-size 100
```

## Tests

```bash
docker compose up -d   # tests need a live Postgres
uv run pytest
```
