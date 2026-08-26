# ResearchBridge

Research intelligence / research-to-impact decision support platform. See
`ResearchBridge.md` for the full architectural blueprint — that document is
the source of truth for all design decisions.

The product center is the **ResearchInput → ResearchAssessment** workflow:
give ResearchBridge a research idea (text) or an uploaded paper, and it
searches the corpus, retrieves related work, compares the input against it,
and returns one evidence-grounded assessment report — novelty, research
gap, technical feasibility, potential applications, and a recommendation,
each field traceable back to real quoted passages. Corpus ingestion,
extraction, embedding/retrieval, and gap detection are supporting
infrastructure for that workflow, not the product itself.

## Setup

```bash
uv sync
docker compose up -d
cp .env.example .env
uv run alembic upgrade head
```

## Backend

FastAPI app (`src/researchbridge/api/app.py`) exposing assessments, corpus
browsing, Q&A, gap review, and admin/pipeline-monitoring routes.

```bash
uv run uvicorn researchbridge.api.app:app --reload
```

### CLI pipeline

Ingestion connectors:

```bash
uv run rb-ingest --search-query "cat:cs.LG OR cat:cs.AI" --page-size 100
uv run rb-ingest-semantic-scholar --query "..."
```

Springer Nature (requires `SPRINGER_META_API_KEY` in `.env` — register at
https://dev.springernature.com; on the free tier, field-scoped queries like
`subject:"..."` and page sizes above 25 both 403 as "premium feature" —
verified live, so the default query is free-text and the default page
size is 25):

```bash
uv run rb-ingest-springer --query '"artificial intelligence" OR "machine learning" OR "computer science"' --page-size 25
```

Extraction, embedding/search, retrieval evaluation, and gap detection:

```bash
uv run rb-extract
uv run rb-extract-evaluate
uv run rb-embed
uv run rb-search --query "..."
uv run rb-retrieval-compare
uv run rb-retrieval-evaluate
uv run rb-gaps-detect --all
```

Benchmark sampling:

```bash
uv run rb-benchmark-sample
uv run rb-benchmark-fetch
```

## Frontend

Next.js app in `frontend/`. `/` is the assessment console (submit an idea
or upload a paper); other routes cover corpus browsing (`/corpus`), paper
detail (`/papers/[id]`), assessment reports (`/assessments/[id]`), Q&A
(`/ask`), gap review (`/gaps`), annotation (`/annotate`), and pipeline/corpus
admin (`/admin`).

```bash
cd frontend
npm install
npm run dev
```

The frontend has no automated test infra — verify changes with a live
browser preview.

## Tests

```bash
docker compose up -d   # tests need a live Postgres
uv run pytest
```
