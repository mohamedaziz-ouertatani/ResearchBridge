# Corpus Curation (Paper Exclusion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator reversibly remove an out-of-domain or noisy paper from every forward-looking part of the system (search, retrieval, gap-detection seeding) without ever hard-deleting a row or breaking the evidence trail of anything already produced.

**Architecture:** A single nullable `papers.excluded_at` timestamp column. `NULL` means included (the default). Setting it filters the paper out of four specific query paths going forward; nothing else in the schema changes, and no existing row referencing this paper (`Evidence`, `CandidateGap`, `ResearchAssessment`, etc.) is touched.

**Tech Stack:** FastAPI + SQLAlchemy + Alembic (backend), Next.js App Router + TypeScript (frontend), pytest against a real Postgres test database (`tests/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-08-21-corpus-curation-design.md`

## Global Constraints

- No hard deletes anywhere in this feature. Soft-delete only (`excluded_at`).
- No `excluded_reason` field — plain timestamp only (spec's explicit scope decision).
- No cascading changes to `Evidence`, `CandidateGap`, `ResearchAssessment`, or `ResearchAssessmentEvidence` — their rows and evidence trails stay exactly as they are regardless of a referenced paper's exclusion state.
- `GET /api/papers/{id}` (single-paper fetch) stays unfiltered by exclusion — only listings and retrieval queries filter.
- No bulk-exclude UI. One paper at a time.
- Frontend has no test infrastructure in this repo — verify frontend tasks live via the Browser pane against the real corpus, not with automated tests.

---

### Task 1: `excluded_at` column (migration + model)

**Files:**
- Create: `migrations/versions/0010_papers_excluded_at.py`
- Modify: `src/researchbridge/db/models.py:21-41` (the `Paper` class)

**Interfaces:**
- Produces: `Paper.excluded_at: datetime | None` — every later task reads or writes this attribute directly on `Paper` ORM instances.

- [ ] **Step 1: Write the migration**

```python
# migrations/versions/0010_papers_excluded_at.py
"""papers: add excluded_at (soft-delete for corpus curation)

NULL = included (default). Set = excluded from search, similar-papers,
ResearchAssessment retrieval, and gap-detection seed selection going
forward. Never a hard delete - see docs/superpowers/specs/
2026-08-21-corpus-curation-design.md for why.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("excluded_at", sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "excluded_at")
```

- [ ] **Step 2: Add the field to the `Paper` model**

In `src/researchbridge/db/models.py`, inside the `Paper` class, add the new column right after `updated_at`:

```python
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    excluded_at: Mapped[datetime | None] = mapped_column(nullable=True)
```

- [ ] **Step 3: Verify the model change is syntactically valid**

Run: `.venv/Scripts/python -c "from researchbridge.db.models import Paper; print(Paper.excluded_at)"`
Expected: prints something like `Paper.excluded_at` with no import/attribute errors.

- [ ] **Step 4: Apply the migration to the real dev database**

Run: `uv run alembic upgrade head`
Expected: output ends with `Running upgrade 0009 -> 0010, papers: add excluded_at (soft-delete for corpus curation)` and no errors. (The test database used by pytest doesn't need this — `tests/conftest.py`'s `engine` fixture calls `Base.metadata.create_all`, which already picks up the new column from the model change in Step 2.)

- [ ] **Step 5: Commit**

```bash
git add migrations/versions/0010_papers_excluded_at.py src/researchbridge/db/models.py
git commit -m "feat: add papers.excluded_at for reversible corpus curation"
```

---

### Task 2: Expose `excluded_at` via the API

**Files:**
- Modify: `src/researchbridge/api/schemas.py:21-33` (`PaperSummary`)
- Modify: `src/researchbridge/api/serializers.py:42-64` (`to_summaries`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Paper.excluded_at` (Task 1).
- Produces: `PaperSummary.excluded_at: datetime | None` — every response that returns a `PaperSummary` (list, get, search, similar) now carries this field. Task 6's exclude endpoint returns this same schema.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (near `test_get_paper_returns_detail`):

```python
def test_paper_summary_includes_excluded_at(client, session) -> None:
    paper = _add_paper(session, "p1")

    body = client.get(f"/api/papers/{paper.id}").json()

    assert body["excluded_at"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_api.py::test_paper_summary_includes_excluded_at -v`
Expected: FAIL with a pydantic/KeyError-style failure — `excluded_at` missing from the response body (the schema doesn't have the field yet).

- [ ] **Step 3: Add the field to `PaperSummary`**

In `src/researchbridge/api/schemas.py`, add an import and the new field:

```python
from datetime import date, datetime
```

```python
class PaperSummary(BaseModel):
    id: uuid.UUID
    source: str
    source_id: str
    title: str
    abstract: str | None
    publication_date: date | None
    url: str | None
    primary_category: str | None
    categories: list[str]
    authors: list[str]
    excluded_at: datetime | None
    """Set when an operator has excluded this paper from search, retrieval,
    and gap-detection seeding (blueprint corpus curation). NULL = included."""

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Pass the field through in the serializer**

In `src/researchbridge/api/serializers.py`, inside `to_summaries`, add `excluded_at=paper.excluded_at` to the `PaperSummary(...)` constructor call:

```python
    return [
        PaperSummary(
            id=paper.id,
            source=paper.source,
            source_id=paper.source_id,
            title=paper.title,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            url=paper.url,
            primary_category=(paper.raw_metadata or {}).get("primary_category"),
            categories=categories_by_paper.get(paper.id, []),
            authors=authors_by_paper.get(paper.id, []),
            excluded_at=paper.excluded_at,
        )
        for paper in papers
    ]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_api.py::test_paper_summary_includes_excluded_at -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/serializers.py tests/test_api.py
git commit -m "feat: expose excluded_at on PaperSummary"
```

---

### Task 3: Filter excluded papers out of similarity search

**Files:**
- Modify: `src/researchbridge/embedding/search.py`
- Test: `tests/test_similarity_search.py`

**Interfaces:**
- Consumes: `Paper.excluded_at` (Task 1).
- Produces: no interface change — `find_similar_to_paper` and `search_by_text` keep their existing signatures. This is the single choke point behind corpus search, similar-papers, `ResearchAssessment` retrieval, and gap-detection's neighborhood lookup — none of those callers need to change.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_similarity_search.py`:

```python
def test_search_by_text_excludes_papers_marked_excluded(session_factory) -> None:
    session = session_factory()
    target = _unit_vector(0)
    included = _make_paper_with_embedding(session, "included", target)
    excluded = _make_paper_with_embedding(session, "excluded", target)
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = search_by_text(session, "query", FixedVectorEmbedder(target), top_k=10)

    result_ids = {paper.id for paper, _distance in results}
    assert included.id in result_ids
    assert excluded.id not in result_ids
    session.close()


def test_find_similar_to_paper_excludes_papers_marked_excluded(session_factory) -> None:
    session = session_factory()
    target = _unit_vector(0)
    seed = _make_paper_with_embedding(session, "seed", target)
    excluded = _make_paper_with_embedding(session, "excluded", target)
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = find_similar_to_paper(session, seed.id, MODEL, top_k=10)

    result_ids = {paper.id for paper, _distance in results}
    assert excluded.id not in result_ids
    session.close()
```

Add the missing import at the top of the file:

```python
from datetime import datetime, timezone
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_similarity_search.py -k excludes_papers_marked_excluded -v`
Expected: both FAIL — the excluded paper still shows up in `result_ids` because nothing filters it out yet.

- [ ] **Step 3: Add the filter in `_nearest`**

In `src/researchbridge/embedding/search.py`, modify `_nearest`:

```python
def _nearest(
    session: Session,
    vector: list[float],
    model_name: str,
    top_k: int,
    exclude_paper_id: uuid.UUID | None,
) -> list[tuple[Paper, float]]:
    distance = Embedding.vector.cosine_distance(vector).label("distance")
    query = (
        select(Paper, distance)
        .join(Embedding, Embedding.paper_id == Paper.id)
        .where(
            Embedding.embedding_type == EMBEDDING_TYPE,
            Embedding.model_name == model_name,
            Paper.excluded_at.is_(None),
        )
    )
    if exclude_paper_id is not None:
        query = query.where(Paper.id != exclude_paper_id)
    query = query.order_by(distance.asc()).limit(top_k)

    return [(row[0], row[1]) for row in session.execute(query).all()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_similarity_search.py -v`
Expected: all PASS (including the two new tests and every pre-existing one in this file).

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/embedding/search.py tests/test_similarity_search.py
git commit -m "feat: exclude excluded papers from similarity search and retrieval"
```

---

### Task 4: `include_excluded` filter on paper browsing

**Files:**
- Modify: `src/researchbridge/api/routes.py:25-48` (`list_papers`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Paper.excluded_at` (Task 1).
- Produces: `GET /api/papers` gains an `include_excluded: bool = False` query parameter. No other endpoint or internal function changes.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
def test_list_papers_hides_excluded_papers_by_default(client, session) -> None:
    included = _add_paper(session, "included")
    excluded = _add_paper(session, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/papers").json()

    ids = {item["id"] for item in body["items"]}
    assert str(included.id) in ids
    assert str(excluded.id) not in ids
    assert body["total"] == 1


def test_list_papers_include_excluded_shows_them(client, session) -> None:
    excluded = _add_paper(session, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/papers?include_excluded=true").json()

    ids = {item["id"] for item in body["items"]}
    assert str(excluded.id) in ids
```

Add the missing import at the top of `tests/test_api.py` if not already present:

```python
from datetime import date, datetime, timezone
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -k "excluded" -v`
Expected: `test_list_papers_hides_excluded_papers_by_default` FAILS (the excluded paper is still in `items`, `total` is 2 not 1). `test_paper_summary_includes_excluded_at` from Task 2 still passes.

- [ ] **Step 3: Add the filter to `list_papers`**

In `src/researchbridge/api/routes.py`, modify `list_papers`:

```python
@router.get("/papers", response_model=PaperPage)
def list_papers(
    session: Session = Depends(get_session),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    year: int | None = Query(None, description="Filter by publication year"),
    category: str | None = Query(None, description="Filter by an arXiv category, e.g. cs.LG"),
    q: str | None = Query(None, description="Case-insensitive substring match on title"),
    include_excluded: bool = Query(False, description="Include papers excluded from curation"),
) -> PaperPage:
    """Browse the corpus. `q` is a plain title filter - for meaning-based search use /api/search."""
    query = select(Paper)
    count_query = select(func.count(Paper.id))

    if not include_excluded:
        query = query.where(Paper.excluded_at.is_(None))
        count_query = count_query.where(Paper.excluded_at.is_(None))

    for condition in _filters(year=year, category=category, q=q):
        query = query.where(condition)
        count_query = count_query.where(condition)

    total = session.execute(count_query).scalar_one()
    papers = list(
        session.execute(query.order_by(Paper.publication_date.desc().nullslast(), Paper.id).limit(limit).offset(offset))
        .scalars()
    )

    return PaperPage(items=to_summaries(session, papers), total=total, limit=limit, offset=offset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: all PASS, including every pre-existing test in this file (the default behavior for callers that never pass `include_excluded` is unchanged for a corpus with no excluded papers).

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/api/routes.py tests/test_api.py
git commit -m "feat: hide excluded papers from corpus browsing by default"
```

---

### Task 5: Exclude excluded papers from gap-detection seed selection

**Files:**
- Modify: `src/researchbridge/gaps/batch.py`
- Test: `tests/test_gaps_batch.py`

**Interfaces:**
- Consumes: `Paper.excluded_at` (Task 1).
- Produces: no interface change — `_select_seed_papers` keeps its existing signature and is only called internally by `run_all`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gaps_batch.py`:

```python
def test_skips_excluded_papers_as_seeds(session_factory, embedder) -> None:
    session = session_factory()
    excluded = _paper(session, embedder, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    summary = run_all(session, embedder, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.papers_seen == 0
```

Add the missing import at the top of `tests/test_gaps_batch.py`:

```python
from datetime import datetime, timezone
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_gaps_batch.py::test_skips_excluded_papers_as_seeds -v`
Expected: FAIL — `summary.papers_seen == 1`, because the excluded paper is still picked up as a seed (it has an embedding, so `_select_seed_papers` currently returns it).

- [ ] **Step 3: Add the filter to `_select_seed_papers`**

In `src/researchbridge/gaps/batch.py`, modify `_select_seed_papers`:

```python
def _select_seed_papers(session: Session, force: bool) -> list[uuid.UUID]:
    query = select(Paper.id).where(
        exists().where(Embedding.paper_id == Paper.id, Embedding.embedding_type == EMBEDDING_TYPE),
        Paper.excluded_at.is_(None),
    )
    if not force:
        already_done = exists().where(CandidateGap.seed_paper_id == Paper.id)
        query = query.where(~already_done)
    return list(session.execute(query).scalars())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_gaps_batch.py -v`
Expected: all PASS, including every pre-existing test in this file.

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/batch.py tests/test_gaps_batch.py
git commit -m "feat: exclude excluded papers from gap-detection seed selection"
```

---

### Task 6: Exclude/include toggle endpoint

**Files:**
- Modify: `src/researchbridge/api/schemas.py` (add `PaperExclude`)
- Modify: `src/researchbridge/api/admin_routes.py`
- Test: `tests/test_admin_api.py`

**Interfaces:**
- Consumes: `PaperSummary` (Task 2), `to_summary` from `researchbridge.api.serializers` (existing function, signature `to_summary(session: Session, paper: Paper) -> PaperSummary`).
- Produces: `PUT /api/admin/papers/{paper_id}/exclude` — request body `{"excluded": bool}`, response `PaperSummary`, `404` for an unknown paper id.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_admin_api.py`:

```python
def test_exclude_sets_excluded_at(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()

    response = client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    assert response.status_code == 200
    assert response.json()["excluded_at"] is not None


def test_exclude_false_clears_excluded_at(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()
    client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    response = client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": False})

    assert response.json()["excluded_at"] is None


def test_exclude_404s_for_unknown_paper(client) -> None:
    response = client.put(f"/api/admin/papers/{uuid.uuid4()}/exclude", json={"excluded": True})

    assert response.status_code == 404


def test_get_paper_still_works_after_exclusion(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1")
    session.commit()
    client.put(f"/api/admin/papers/{paper.id}/exclude", json={"excluded": True})

    response = client.get(f"/api/papers/{paper.id}")

    assert response.status_code == 200
    assert response.json()["excluded_at"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_admin_api.py -k exclude -v`
Expected: the first three FAIL with 404 (the route doesn't exist yet). `test_get_paper_still_works_after_exclusion` also FAILS because the PUT it depends on 404s.

- [ ] **Step 3: Add the `PaperExclude` request schema**

In `src/researchbridge/api/schemas.py`, add near `ResearchAssessmentReview`:

```python
class PaperExclude(BaseModel):
    excluded: bool
```

- [ ] **Step 4: Add the endpoint**

In `src/researchbridge/api/admin_routes.py`, replace the existing import block (everything from `from __future__ import annotations` down to the `router = ...` line) with:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_session
from researchbridge.api.schemas import PaperExclude, PaperSummary, PipelineRunOut, PipelineStatus
from researchbridge.api.serializers import to_summary
from researchbridge.db.models import (
    Embedding,
    EmbeddingRun,
    ExtractedClaim,
    ExtractionRun,
    IngestionRun,
    Paper,
)

router = APIRouter(prefix="/api/admin")

RECENT_RUNS_LIMIT = 10
```

(`HTTPException`, `uuid`, `datetime`/`timezone`, `PaperExclude`, `PaperSummary`, and `to_summary` are new; everything else already exists in this file. Keep the existing `pipeline_status` route and its `_recent`/`_to_run` helpers exactly as they are — just add the following new route below them, at the end of the file.)

```python
@router.put("/papers/{paper_id}/exclude", response_model=PaperSummary)
def exclude_paper(
    paper_id: uuid.UUID, payload: PaperExclude, session: Session = Depends(get_session)
) -> PaperSummary:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")

    paper.excluded_at = datetime.now(timezone.utc) if payload.excluded else None
    session.commit()

    return to_summary(session, paper)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_admin_api.py -v`
Expected: all PASS, including every pre-existing test in this file.

- [ ] **Step 6: Run the full backend suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests pass (this is the last backend task — confirms nothing upstream broke across the whole feature).

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/admin_routes.py tests/test_admin_api.py
git commit -m "feat: add paper exclude/include toggle endpoint"
```

---

### Task 7: Frontend — exclude toggle on the paper detail page

**Files:**
- Modify: `frontend/lib/api.ts` (`PaperSummary` type)
- Modify: `frontend/lib/adminApi.ts`
- Modify: `frontend/app/papers/[id]/page.tsx`

**Interfaces:**
- Consumes: `PUT /api/admin/papers/{id}/exclude` (Task 6).
- Produces: `adminApi.excludePaper(id: string, excluded: boolean): Promise<PaperSummary>` — no other frontend file needs this in this task, but Task 8 imports the same `PaperSummary.excluded_at` field.

- [ ] **Step 1: Add `excluded_at` to the shared `PaperSummary` type**

In `frontend/lib/api.ts`, modify the `PaperSummary` type:

```typescript
export type PaperSummary = {
  id: string;
  source: string;
  source_id: string;
  title: string;
  abstract: string | null;
  publication_date: string | null;
  url: string | null;
  primary_category: string | null;
  categories: string[];
  authors: string[];
  excluded_at: string | null;
};
```

- [ ] **Step 2: Add the exclude call to `adminApi.ts`**

In `frontend/lib/adminApi.ts`, add the import and the new function:

```typescript
import { API_BASE, type PaperSummary } from "./api";
```

```typescript
export const adminApi = {
  pipelineStatus: () =>
    fetch(`${API_BASE}/api/admin/pipeline`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<PipelineStatus>;
    }),

  excludePaper: (id: string, excluded: boolean) =>
    fetch(`${API_BASE}/api/admin/papers/${id}/exclude`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ excluded }),
      cache: "no-store",
    }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<PaperSummary>;
    }),
};
```

- [ ] **Step 3: Add the toggle to the paper detail page**

In `frontend/app/papers/[id]/page.tsx`, add the import:

```typescript
import { adminApi } from "@/lib/adminApi";
```

Add a small toggle component below the imports, and use it in the header block. Replace the categories/date block:

```typescript
function ExcludeToggle({ paper, onChange }: { paper: PaperSummary; onChange: (p: PaperSummary) => void }) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const excluded = paper.excluded_at !== null;

  async function toggle() {
    setBusy(true);
    setFailed(false);
    try {
      onChange(await adminApi.excludePaper(paper.id, !excluded));
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="inline-flex items-center gap-2">
      <button
        onClick={toggle}
        disabled={busy}
        className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] disabled:opacity-50 ${
          excluded
            ? "border-[var(--live)] text-[var(--live)] hover:opacity-80"
            : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
        }`}
      >
        {busy ? "saving…" : excluded ? "excluded — include again" : "exclude this paper"}
      </button>
      {failed && <span className="text-[0.6875rem] text-[var(--live)]">save failed</span>}
    </span>
  );
}
```

In the same file, update the `PaperDetail` component to pass `paper` state down and render the toggle in the existing categories row:

```typescript
          <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="readout text-[0.75rem] text-[var(--ink-faint)]">
              {paper.source}:{paper.source_id}
            </span>
            {paper.categories.map((category) => (
              <span
                key={category}
                className="readout rounded-[2px] bg-[var(--field)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]"
              >
                {category}
              </span>
            ))}
            <ExcludeToggle paper={paper} onChange={setPaper} />
          </div>
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output (clean).

- [ ] **Step 5: Verify live in the Browser pane**

Start the `api` and `frontend` dev servers (per `.claude/launch.json`), navigate to a real paper's detail page (e.g. `/papers/<any-real-id-from-the-corpus>`), click "exclude this paper", confirm the button flips to "excluded — include again", reload the page, confirm the state persisted, click again to include it back and confirm it flips back.

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api.ts frontend/lib/adminApi.ts frontend/app/papers/[id]/page.tsx
git commit -m "feat: add exclude toggle to the paper detail page"
```

---

### Task 8: Frontend — "show excluded" filter on the corpus page

**Files:**
- Modify: `frontend/lib/api.ts` (`api.papers` params)
- Modify: `frontend/components/PaperRow.tsx`
- Modify: `frontend/app/corpus/page.tsx`

**Interfaces:**
- Consumes: `PaperSummary.excluded_at` (Task 7), `GET /api/papers?include_excluded=` (Task 4).
- Produces: no new exports — this is the last task in the feature.

- [ ] **Step 1: Add `include_excluded` to the `papers` call**

In `frontend/lib/api.ts`, modify the `papers` function in the `api` object:

```typescript
  papers: (params: {
    limit?: number;
    offset?: number;
    year?: number;
    category?: string;
    q?: string;
    include_excluded?: boolean;
  }) => get<PaperPage>("/api/papers", params),
```

- [ ] **Step 2: Show an "excluded" marker on `PaperRow`**

In `frontend/components/PaperRow.tsx`, add the marker next to the existing category tag:

```typescript
            {paper.primary_category && (
              <span className="readout rounded-[2px] bg-[var(--field)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]">
                {paper.primary_category}
              </span>
            )}
            {paper.excluded_at && (
              <span className="readout rounded-[2px] border border-[var(--live)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--live)]">
                excluded
              </span>
            )}
```

- [ ] **Step 3: Add the filter chip to `/corpus`**

In `frontend/app/corpus/page.tsx`, add state and wire it into `loadBrowse`:

```typescript
  const [showExcluded, setShowExcluded] = useState(false);
```

```typescript
  const loadBrowse = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const page = await api.papers({
        limit: 25,
        year: year ?? undefined,
        category: category ?? undefined,
        include_excluded: showExcluded || undefined,
      });
      setPapers(page.items);
      setTotal(page.total);
    } catch {
      setError("Couldn't load papers.");
    } finally {
      setBusy(false);
    }
  }, [year, category, showExcluded]);
```

Add the chip itself near the existing category filter buttons, inside the `{stats && mode === "browse" && (...)}` block, right after the categories `<div>`:

```typescript
          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={() => setShowExcluded((v) => !v)}
              aria-pressed={showExcluded}
              className={`readout rounded-[2px] border px-2 py-1 text-[0.6875rem] transition-colors ${
                showExcluded
                  ? "border-[var(--live)] text-[var(--live)]"
                  : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)]"
              }`}
            >
              show excluded
            </button>
          </div>
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output (clean).

- [ ] **Step 5: Verify live in the Browser pane**

Navigate to `/corpus`, confirm the "show excluded" chip is present and off by default, exclude a paper via its detail page (Task 7), return to `/corpus`, confirm it's no longer in the default browse results, toggle "show excluded" on, confirm it reappears with the "excluded" marker from Step 2, toggle back off, confirm it disappears again.

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api.ts frontend/components/PaperRow.tsx frontend/app/corpus/page.tsx
git commit -m "feat: add show-excluded filter to the corpus browser"
```
