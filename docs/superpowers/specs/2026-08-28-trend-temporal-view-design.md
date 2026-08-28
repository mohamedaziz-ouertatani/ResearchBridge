# Trend/Temporal View — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint), §32's "temporal patterns" pipeline step. That step is aspirational text only — nothing in `gaps/detect.py` or `gaps/cluster.py` implements it, and no existing page or endpoint does corpus-wide temporal analysis. This spec covers a real, defensible version of it: counts of already-extracted claims over time within one category, not semantic pattern inference.

## Scope

In scope:
- A new endpoint, `GET /api/trends?category=<category>`, returning per-year counts of each `claim_type` for papers in that category — a direct count query, no clustering or inference.
- A new page, `app/trends/page.tsx`, with a category picker and one bar-strip chart per claim_type, all sharing a common year axis.
- Added to primary navigation (`Nav.tsx`) alongside `/corpus` and `/assessments`.

Out of scope (deliberately deferred):
- Semantic clustering of papers within a category/year to detect recurring *themes* (as opposed to counting existing claim types). This would reintroduce the gap-engine's clustering machinery (threshold tuning, minimum cluster size) for a second, competing purpose — a future extension once counted trends prove useful, not part of this build.
- An "all categories" or cross-category comparison view. Category is a required query parameter; there is no unscoped mode.
- A new charting library or new CSS. Reuses the `.tickstrip`/`.tick`/`.tick-bar` bar-strip pattern `AdminStats.tsx`'s `IngestionVolume` already established, and the zero-filled-year-range convention `AssessmentReport.tsx`'s `IdeaYearTrend` already established for its per-assessment chart.
- Excluding low-confidence claims from the counts. A claim's `confidence` ("high"/"medium"/"low") describes the extractor's certainty about that one claim's text, not whether the claim happened at all — a "low confidence, verbatim uncertain" limitation claim is still a real extracted data point for volume-over-time purposes. Only `extraction_method="stub"` rows (synthetic test data) and curated-out papers are excluded, same as every other claim-facing endpoint.
- Frontend automated tests. No test infra exists yet; manually verified in-browser, consistent with every other frontend feature this session.

## Data flow

```
GET /api/trends?category=cs.LG
  -> join Paper -> PaperCategory (category = "cs.LG") -> ExtractedClaim -> Evidence
     WHERE Paper.excluded_at IS NULL AND Evidence.extraction_method != 'stub'
  -> GROUP BY extract(year, Paper.publication_date), ExtractedClaim.claim_type
  -> COUNT(*)
  -> zero-fill every year in [min(year), max(year)] observed for this category,
     for every claim_type present in the result set (same convention as
     AssessmentReport.tsx's IdeaYearTrend: gaps are meaningful, not skipped)
  -> return {category, years: [...], series: {claim_type: [counts aligned to years]}}
```

If the category has no non-stub claims at all, `years` is `[]` and `series` is `{}` — the frontend renders its empty state rather than an empty chart.

## Backend

**New schema** in `schemas.py` (near `CorpusStats`):

```python
class TrendsOut(BaseModel):
    category: str
    years: list[int]
    series: dict[str, list[int]]
    """claim_type -> counts, each list the same length as `years` and aligned
    to it by index. Only claim_types with at least one non-stub claim in this
    category appear as keys."""
```

**New route** in `routes.py`, alongside `corpus_stats` (no new module — this is one query, same precedent `corpus_stats` itself sets for not needing a dedicated file):

```python
@router.get("/trends", response_model=TrendsOut)
def trends(
    category: str = Query(..., description="An ingested category, e.g. cs.LG"),
    session: Session = Depends(get_session),
) -> TrendsOut:
    ...
```

Query shape: one `SELECT extract(year, ...), claim_type, count(*)` grouped and filtered as above, executed once. No parameter beyond `category` — no year-range filter, since the whole point is showing the full observed range.

## Frontend

- `frontend/lib/api.ts`: add `trends: (category: string) => get<TrendsData>('/api/trends', { category })` and the matching `TrendsData`/type.
- `frontend/app/trends/page.tsx`:
  - Category `<select>` populated from `api.stats()`'s existing `papers_by_category` (no new endpoint needed for the picker) — same `<select>` pattern the assessments list page's novelty/feasibility filters just established.
  - On category change, fetch `/api/trends` and render one `TrendStrip` per claim_type (a thin wrapper around the existing `.tickstrip`/`.tick`/`.tick-bar` classes, parameterized by label + counts + a shared year axis) — modeled directly on `AdminStats.tsx`'s `IngestionVolume`, not a new charting primitive.
  - Claim types render in a fixed, stable order (the same order `extracted_claims.claim_type` values are documented in the blueprint: problem, method, dataset, metric, result, limitation, research_gap, application, contribution) rather than whatever order the API returns keys in, so the page doesn't reflow between categories.
  - Empty state: "No extracted claims yet for this category" when `years` is empty.
- `frontend/components/Nav.tsx`: add a `/trends` link alongside `/corpus` and `/assessments`.

## Error handling

- `category` missing → FastAPI's standard 422 for a required query param, same as every other required `Query(...)` in this codebase.
- `category` not matching any ingested category → not an error; empty `years`/`series`, handled by the frontend's empty state (mirrors `/api/trends`'s own "nothing here" case, not a distinct "unknown category" error — there's no fixed enum of valid categories to validate against, only whatever the corpus happens to contain).
- Frontend fetch failure → inline error scoped to the page, same pattern every other page in this app already uses.

## Testing

- Backend: route test (`GET /api/trends?category=...`) covering: correct per-year/claim_type counts from a small fixture corpus, zero-fill across the observed year range, stub claims excluded, excluded (curated-out) papers excluded, missing `category` param returns 422, unknown category returns empty `years`/`series`.
- Frontend: manually verified in-browser (dev server, category switch, empty-state check), consistent with every other frontend feature this session's lack of automated tests.
