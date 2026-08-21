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

Springer Nature (requires `SPRINGER_META_API_KEY` in `.env` — register at
https://dev.springernature.com; on the free tier, field-scoped queries like
`subject:"..."` and page sizes above 25 both 403 as "premium feature" —
verified live, so the default query is free-text and the default page
size is 25):

```bash
uv run rb-ingest-springer --query '"artificial intelligence" OR "machine learning" OR "computer science"' --page-size 25
```

## Tests

```bash
docker compose up -d   # tests need a live Postgres
uv run pytest
```
