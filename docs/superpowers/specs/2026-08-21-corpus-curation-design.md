# Corpus Curation (Paper Exclusion) — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint). This spec covers the second of two admin-panel sub-projects (the first, read-only pipeline monitoring, shipped in commit `9886c75`). This slice: a reversible way to remove an out-of-domain or noisy paper from everything the system does going forward, without breaking anything already built on top of it.

## Scope

In scope:
- A soft-delete flag (`papers.excluded_at`) — never a hard row delete. Reversible.
- Filtering excluded papers out of every forward-looking retrieval path: corpus browse, semantic search, similar-papers, `ResearchAssessment` retrieval, and gap-detection seed selection.
- One toggle endpoint (`PUT /api/admin/papers/{id}/exclude`) and one UI toggle on the existing paper detail page.
- A `include_excluded` filter on the existing corpus browse endpoint/page, so an excluded paper can still be found and un-excluded.

Out of scope (deliberately deferred):
- Hard deletion of any row. No cascading deletes anywhere in this slice.
- An `excluded_reason` field — plain boolean/timestamp only, per explicit scope decision.
- A dedicated curation/moderation list page — the toggle lives on the existing paper detail page instead.
- Retroactively touching `CandidateGap`, `ResearchAssessment`, `Evidence`, or `ResearchAssessmentEvidence` rows that already reference an excluded paper. Their evidence trail must stay intact and inspectable exactly as it was.
- Bulk exclude (multi-select). One paper at a time, matching the "lightweight, not a dashboard" framing of the whole admin panel.

## Why soft-delete, not hard delete

No table currently has `ON DELETE CASCADE` configured, so a raw row delete on `papers` fails immediately on the first foreign-key constraint (`paper_authors`, `evidence`, etc. all reference `papers.id` with `nullable=False`). More importantly, this project's central design principle — every `ResearchAssessment` field and every `CandidateGap` observation must trace to real, still-inspectable `Evidence` (§15/§17, carried through the whole assessment build) — would be violated by deleting a paper's `Evidence` rows out from under an already-completed assessment or an already-approved candidate gap. A soft-delete flag that only affects *future* retrieval avoids this entirely: nothing already produced ever loses its grounding.

## Data model

One migration, additive only:

```
papers
------
...(all existing columns unchanged)...
excluded_at         TIMESTAMPTZ NULL   -- NULL = included (default). Set = excluded from all forward retrieval.
```

No new tables. No changes to any other table.

## Retrieval filtering

Four choke points, chosen because they're the only places papers enter a query result that could feed a *new* action (as opposed to displaying already-produced history):

1. **`embedding/search.py::_nearest`** — the single shared query behind both `search_by_text` and `find_similar_to_paper`. Adding `Paper.excluded_at.is_(None)` here covers, in one place:
   - `GET /api/search` (corpus semantic search)
   - `GET /api/papers/{id}/similar`
   - `assessment/build.py`'s retrieval step (every new `ResearchAssessment`)
   - `gaps/detect.py`'s neighborhood lookup (every new gap-detection run)
2. **`api/routes.py::list_papers`** — excluded papers hidden from `/api/papers` (and therefore `/corpus` browsing) by default. A new `include_excluded: bool = False` query param opts back in, so curation stays possible.
3. **`gaps/batch.py::_select_seed_papers`** — excluded papers are never chosen as a seed for `rb-gaps-detect --all`.
4. **`api/routes.py::get_paper`** (single-paper fetch by id) — deliberately **unfiltered**. The paper detail page must still load an excluded paper (to show its state and let it be un-excluded); only listings and retrieval are filtered.

Nothing else changes. `PaperSummary` (returned by list/get/search/similar) gains one field: `excluded_at: datetime | None`, so the frontend can render current state without a second request.

## API

`PUT /api/admin/papers/{id}/exclude`, mirroring the `ResearchAssessment` review-toggle pattern (`assessment_routes.py::review_assessment`) exactly:

```
Request:  {"excluded": bool}
Response: PaperSummary (excluded_at set to now() if excluded=true, cleared to null if excluded=false)
404 if the paper id doesn't exist.
```

Lives in `admin_routes.py` (extends the router already created for pipeline monitoring) rather than `routes.py`, since it's a curation action, not a browse/search endpoint.

`GET /api/papers` gains `include_excluded: bool = Query(False)`.

## Frontend

- **`/papers/[id]`**: an "exclude this paper" / "excluded — include again" toggle, placed near the existing category tags in the header block. Same local busy/failed state pattern as `AssessmentReport.tsx`'s review toggle (`useState` + try/catch around the API call, no optimistic update beyond what the response confirms).
- **`/corpus`**: a small "show excluded" filter chip alongside the existing year/category filters, toggling `include_excluded` on the `/api/papers` call. When on, excluded papers render with a visible "excluded" marker in `PaperRow` so they're not confused with normal results.

## Testing

- Backend: pytest against the real `IngestionRun`-style test-DB fixtures already used throughout — exclude/include round-trip, 404 on unknown id, and one test per retrieval choke point confirming an excluded paper is absent from that query's results (search, similar, list, gap-detection seed selection) while still fetchable directly by id.
- Frontend: no test infra in this repo (consistent with every other frontend feature this session) — verified live via browser-preview against the real corpus: exclude a real paper, confirm it drops out of `/corpus` and `/api/search`, confirm `include_excluded=true` brings it back, confirm un-excluding restores it everywhere.

## Migration safety

Additive-only column with a `NULL` default — no backfill needed, no existing row's behavior changes until an admin explicitly excludes something. Purely reversible: setting `excluded_at` back to `NULL` fully restores a paper to every retrieval path with no data loss, because nothing was ever deleted.
