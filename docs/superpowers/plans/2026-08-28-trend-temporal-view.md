# Trend/Temporal View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a corpus-wide trend view showing per-year counts of each extracted claim_type, scoped to one category at a time.

**Architecture:** One new backend route (`GET /api/trends`) does a single grouped-count SQL query over `Paper`/`PaperCategory`/`ExtractedClaim`/`Evidence`, zero-fills the observed year range, and returns counts per claim_type. One new frontend page (`/trends`) renders a category picker (reusing the existing `/api/stats` category list) and one bar-strip chart per claim_type, reusing the `.tickstrip`/`.tick`/`.tick-bar` CSS classes `AdminStats.tsx`'s `IngestionVolume` already established — no new charting code or library.

**Tech Stack:** Python (SQLAlchemy), FastAPI/Pydantic, Next.js/TypeScript.

**Spec:** `docs/superpowers/specs/2026-08-28-trend-temporal-view-design.md`

## Global Constraints

- `category` is a required query parameter — no "all categories" mode.
- Excludes `Evidence.extraction_method="stub"` claims and `Paper.excluded_at`-set papers, same as every other claim-facing endpoint in this codebase.
- Does NOT filter by claim confidence (`high`/`medium`/`low`) — confidence describes per-claim certainty, not whether the claim counts at all.
- No semantic clustering, no new charting library, no "all categories" comparison view.
- Zero-fill every year in `[min(observed year), max(observed year)]` for this category, even years with no claims at all in that range — same convention `AssessmentReport.tsx`'s `IdeaYearTrend` already established.
- No frontend automated tests (no test infra exists yet). Manually verified in-browser.

---

### Task 1: Backend `/api/trends` endpoint

**Files:**
- Modify: `src/researchbridge/api/schemas.py`
- Modify: `src/researchbridge/api/routes.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `TrendsOut` schema (`category: str`, `years: list[int]`, `series: dict[str, list[int]]`); `GET /api/trends?category=<str>` route returning `TrendsOut`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`, after `test_stats_excludes_excluded_papers_by_default`:

```python
def test_trends_returns_counts_by_year_and_claim_type(client, session) -> None:
    p1 = _add_paper(session, "p1", year=2019, categories=("cs.LG",))
    _add_claim(session, p1, "limitation", "evaluated offline only")
    p2 = _add_paper(session, "p2", year=2019, categories=("cs.LG",))
    _add_claim(session, p2, "method", "a transformer architecture")
    p3 = _add_paper(session, "p3", year=2024, categories=("cs.LG",))
    _add_claim(session, p3, "limitation", "small dataset")

    body = client.get("/api/trends", params={"category": "cs.LG"}).json()

    assert body["category"] == "cs.LG"
    assert body["years"] == [2019, 2020, 2021, 2022, 2023, 2024]
    assert body["series"]["limitation"] == [1, 0, 0, 0, 0, 1]
    assert body["series"]["method"] == [1, 0, 0, 0, 0, 0]


def test_trends_only_counts_claims_in_the_requested_category(client, session) -> None:
    lg_paper = _add_paper(session, "p1", year=2020, categories=("cs.LG",))
    _add_claim(session, lg_paper, "limitation", "evaluated offline only")
    cl_paper = _add_paper(session, "p2", year=2020, categories=("cs.CL",))
    _add_claim(session, cl_paper, "limitation", "small test set")

    body = client.get("/api/trends", params={"category": "cs.LG"}).json()

    assert body["series"]["limitation"] == [1]


def test_trends_excludes_stub_claims(client, session) -> None:
    paper = _add_paper(session, "p1", year=2020, categories=("cs.LG",))
    _add_claim(session, paper, "limitation", "synthetic placeholder", extraction_method="stub")

    body = client.get("/api/trends", params={"category": "cs.LG"}).json()

    assert body["years"] == []
    assert body["series"] == {}


def test_trends_excludes_excluded_papers(client, session) -> None:
    paper = _add_paper(session, "p1", year=2020, categories=("cs.LG",))
    _add_claim(session, paper, "limitation", "evaluated offline only")
    paper.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/trends", params={"category": "cs.LG"}).json()

    assert body["years"] == []
    assert body["series"] == {}


def test_trends_requires_category_param(client) -> None:
    assert client.get("/api/trends").status_code == 422


def test_trends_unknown_category_returns_empty(client) -> None:
    body = client.get("/api/trends", params={"category": "cs.NOPE"}).json()

    assert body == {"category": "cs.NOPE", "years": [], "series": {}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k trends -v`
Expected: FAIL with 404 (route doesn't exist yet) on all six tests

- [ ] **Step 3: Write minimal implementation**

In `src/researchbridge/api/schemas.py`, add (near `CorpusStats`):

```python
class TrendsOut(BaseModel):
    category: str
    years: list[int]
    series: dict[str, list[int]]
    """claim_type -> counts, each list the same length as `years` and aligned
    to it by index. Only claim_types with at least one non-stub claim in this
    category appear as keys."""
```

In `src/researchbridge/api/routes.py`:

1. Update the model import line to add `Evidence`:

```python
from researchbridge.db.models import Author, Embedding, Evidence, ExtractedClaim, Paper, PaperAuthor, PaperCategory, PaperCitation
```

2. Update the schema import line to add `TrendsOut` (the current import block already lists `CitationEdgeOut`, `CitationGraphOut`, `CitationNodeOut`, `CorpusStats`, `ExtractedClaimOut`, `PaperPage`, `PaperSummary`, `SearchHit` from the citation-graph-exploration work — just add `TrendsOut` alphabetically into that same block):

```python
from researchbridge.api.schemas import (
    CitationEdgeOut,
    CitationGraphOut,
    CitationNodeOut,
    CorpusStats,
    ExtractedClaimOut,
    PaperPage,
    PaperSummary,
    SearchHit,
    TrendsOut,
)
```

3. Add the route (after `corpus_stats`):

```python
@router.get("/trends", response_model=TrendsOut)
def trends(
    category: str = Query(..., description="An ingested category, e.g. cs.LG"),
    session: Session = Depends(get_session),
) -> TrendsOut:
    """Real counts of already-extracted claims per year within one category -
    not an inferred pattern. See TrendsOut for the response shape."""
    rows = session.execute(
        select(
            func.extract("year", Paper.publication_date),
            ExtractedClaim.claim_type,
            func.count(ExtractedClaim.id),
        )
        .join(PaperCategory, PaperCategory.paper_id == Paper.id)
        .join(ExtractedClaim, ExtractedClaim.paper_id == Paper.id)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            PaperCategory.category == category,
            Evidence.extraction_method != "stub",
            Paper.publication_date.isnot(None),
            Paper.excluded_at.is_(None),
        )
        .group_by(func.extract("year", Paper.publication_date), ExtractedClaim.claim_type)
    ).all()

    if not rows:
        return TrendsOut(category=category, years=[], series={})

    observed_years = sorted({int(year) for year, _, _ in rows})
    years = list(range(observed_years[0], observed_years[-1] + 1))
    year_index = {year: i for i, year in enumerate(years)}

    series: dict[str, list[int]] = {}
    for year, claim_type, count in rows:
        counts = series.setdefault(claim_type, [0] * len(years))
        counts[year_index[int(year)]] = count

    return TrendsOut(category=category, years=years, series=series)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -k trends -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest -q`
Expected: all tests pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/routes.py tests/test_api.py
git commit -m "feat: add GET /api/trends endpoint"
```

---

### Task 2: Frontend API client for the trends endpoint

**Files:**
- Modify: `frontend/lib/api.ts`

**Interfaces:**
- Produces: `TrendsData` type (`category: string`, `years: number[]`, `series: Record<string, number[]>`); `api.trends(category: string): Promise<TrendsData>`.

- [ ] **Step 1: Add the type and API call**

In `frontend/lib/api.ts`, add near the other type definitions (after `CorpusStats`):

```typescript
export type TrendsData = {
  category: string;
  years: number[];
  series: Record<string, number[]>;
};
```

Add to the `api` object, alongside `stats`:

```typescript
  trends: (category: string) => get<TrendsData>("/api/trends", { category }),
```

- [ ] **Step 2: Verify with a typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors (pure addition, nothing else references it yet)

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat: add trends() to the frontend API client"
```

---

### Task 3: Trends page

**Files:**
- Create: `frontend/app/trends/page.tsx`

**Interfaces:**
- Consumes: `api.stats`, `api.trends`, `CorpusStats`, `TrendsData` (Task 2).

- [ ] **Step 1: Write the page**

Create `frontend/app/trends/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats, type TrendsData } from "@/lib/api";
import { Nav } from "@/components/Nav";

// Fixed order matching the blueprint's documented claim_type list, not
// whatever order the API returns keys in - keeps the page from reflowing
// as you switch categories.
const CLAIM_TYPE_ORDER = [
  "problem",
  "method",
  "dataset",
  "metric",
  "result",
  "limitation",
  "research_gap",
  "application",
  "contribution",
];

const SELECT_CLASS =
  "eyebrow rounded-[2px] border border-[var(--rule)] bg-transparent px-2 py-1 text-[0.6875rem] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none";

export default function TrendsPage() {
  const [categories, setCategories] = useState<string[]>([]);
  const [category, setCategory] = useState<string>("");
  const [data, setData] = useState<TrendsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stats()
      .then((stats: CorpusStats) => {
        const sorted = Object.keys(stats.papers_by_category).sort(
          (a, b) => stats.papers_by_category[b] - stats.papers_by_category[a],
        );
        setCategories(sorted);
        if (sorted.length > 0) setCategory(sorted[0]);
      })
      .catch(() => setError("Couldn't load categories."));
  }, []);

  useEffect(() => {
    if (!category) return;
    // Resetting data/error before a fetch keyed on `category` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    api
      .trends(category)
      .then(setData)
      .catch(() => setError("Couldn't load trends."));
  }, [category]);

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">trends</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          How many extracted claims of each type appear per year, within one category - a real
          count of what&apos;s already in the corpus, not an inferred pattern.
        </p>

        <label className="mt-6 flex items-center gap-2">
          <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">category</span>
          <select value={category} onChange={(e) => setCategory(e.target.value)} className={SELECT_CLASS}>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

        {!error && !data && <p className="eyebrow py-16">loading…</p>}

        {!error && data && data.years.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            No extracted claims yet for this category.
          </p>
        )}

        {!error && data && data.years.length > 0 && (
          <div className="mt-8 space-y-6">
            {CLAIM_TYPE_ORDER.filter((claimType) => data.series[claimType]).map((claimType) => (
              <TrendStrip
                key={claimType}
                label={claimType.replace("_", " ")}
                years={data.years}
                counts={data.series[claimType]}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

function TrendStrip({ label, years, counts }: { label: string; years: number[]; counts: number[] }) {
  const peak = Math.max(...counts, 1);

  return (
    <div>
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{label}</span>
      <div className="tickstrip mt-2 h-10" style={{ ["--cols" as string]: years.length }}>
        {years.map((year, i) => (
          <div key={year} className="tick" title={`${year} — ${counts[i].toLocaleString()}`}>
            <span
              className="tick-bar"
              style={{ height: `${Math.max((counts[i] / peak) * 100, counts[i] > 0 ? 4 : 0)}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors. This page isn't linked from navigation yet (Task 4), so this step only confirms it compiles and can be reached by URL.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/trends/page.tsx
git commit -m "feat: add the trends page"
```

---

### Task 4: Wire trends into navigation

**Files:**
- Modify: `frontend/components/Nav.tsx`

**Interfaces:**
- Consumes: `/trends` route (Task 3).

- [ ] **Step 1: Add the nav entry**

In `frontend/components/Nav.tsx`, add `/trends` to the `INFRASTRUCTURE` array, right after `/corpus`:

```typescript
const INFRASTRUCTURE = [
  { href: "/corpus", label: "corpus" },
  { href: "/trends", label: "trends" },
  { href: "/gaps", label: "gap review" },
  { href: "/annotate", label: "annotation workbench" },
  { href: "/admin", label: "pipeline status" },
];
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit` from `frontend/`
Expected: no errors

- [ ] **Step 3: Manually verify in the browser**

Start the dev server (`npm run dev` from `frontend/`). Click "infrastructure" in the nav, then "trends":
- The category picker shows real categories from the corpus, largest first.
- Switching categories reloads the bar strips.
- A category with extracted claims shows one bar-strip row per claim_type present, each bar's height proportional to that year's count, hovering a bar shows its year and count in a tooltip.
- A category with no extracted claims (if one exists in the corpus) shows "No extracted claims yet for this category."

- [ ] **Step 4: Commit**

```bash
git add frontend/components/Nav.tsx
git commit -m "feat: link the trends page from navigation"
```
