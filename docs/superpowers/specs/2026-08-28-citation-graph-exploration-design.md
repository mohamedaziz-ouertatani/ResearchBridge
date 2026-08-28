# Citation Graph Exploration — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint), §12 ("Citation Relationships Without Neo4j") and §31 ("Citation Analysis"). This feature has two parts that must ship together: `paper_citations` is currently schema-only and empty (see its docstring in `db/models.py`) — no connector has ever populated it, so a citation graph view has nothing to show until ingestion exists. Part 1 populates the table; Part 2 visualizes it per-paper, as a sibling to the existing per-assessment similarity graph (`docs/superpowers/specs/2026-08-26-assessment-similarity-graph-design.md`).

## Scope

In scope:
- A new citation-ingestion pass, `rb-citations-fetch`, that fetches citation/reference edges from the Semantic Scholar Graph API's per-paper endpoint for every already-ingested `source="semantic_scholar"` paper, and persists an edge only when **both** the citing and cited paper already exist in the corpus (matched by `source="semantic_scholar", source_id=<paperId>`). Citations to papers outside the corpus are silently skipped, not stubbed.
- A new endpoint, `GET /api/papers/{paper_id}/citations`, returning that paper's direct citation edges (papers it cites, papers that cite it) within the corpus.
- A new frontend component, `CitationGraph`, rendered on the existing paper detail page (`app/papers/[id]/page.tsx`), reusing the `react-force-graph` machinery `SimilarityGraph` already established — the paper itself as the center node, cited/citing papers as satellites, direction distinguished by edge color rather than by distance.

Out of scope (deliberately deferred):
- CrossRef or any other citation source. Semantic Scholar is the only connector with a citations-capable API today; a second source is a future extension once this one is validated, same "prove the first source before adding a second" discipline the blueprint applies to paper ingestion itself (§4).
- External/unresolved citation stubs (citing/cited papers not in the corpus). `paper_citations` keeps its current schema — both FKs `NOT NULL` — no migration needed. Most Semantic Scholar citation lists will point outside this corpus; that's expected and acceptable, not a bug to work around.
- Multi-hop citation traversal (lineages, generations). This spec covers one paper's *direct* edges only. §12's recursive-CTE traversal stays a future capability, introduced only if a real need for it shows up (same "don't build until measured need" rule the blueprint states explicitly).
- An admin-panel trigger or a `*_runs` tracking table for citation fetching. This is a derived/background enrichment pass over the existing corpus, the same tier as `rb-gaps-detect` (CLI-only, dry-run by default, `--save` to persist) — not a live-monitored ingestion source like arXiv/Springer/Semantic Scholar/CORE ingestion.
- Frontend automated tests. No test infra exists on the frontend yet; this stays manually verified in-browser, consistent with `SimilarityGraph`.

## Data flow (ingestion)

```
rb-citations-fetch <arxiv-style source_id> | --all
  -> load target Paper row(s) where source="semantic_scholar" (single paper, or every
     such paper without existing outgoing PaperCitation edges when --all and not --force)
  -> for each paper:
       GET https://api.semanticscholar.org/graph/v1/paper/{paper.source_id}
           ?fields=citations.paperId,references.paperId
       (same throttle/retry pattern as connectors/semantic_scholar.py: 6.0s interval,
        retry on 429/5xx, optional SEMANTIC_SCHOLAR_API_KEY header)
     -> collect citation paperIds (papers that cite this one) and reference paperIds
        (papers this one cites)
     -> resolve each paperId against existing Paper rows (source="semantic_scholar",
        source_id=paperId) in one batched SELECT per paper
     -> for each resolved pair, insert a PaperCitation row (citing_paper_id,
        cited_paper_id, source="semantic_scholar", confidence="high") if it doesn't
        already exist (unique constraint on citing_paper_id/cited_paper_id/source
        already enforces this - insert is try/skip-on-IntegrityError, matching
        ingestion/pipeline.py's _persist pattern)
  -> print a summary: papers processed, edges found, edges skipped (already existed),
     citations/references that didn't resolve to an in-corpus paper
```

Confidence is `"high"`, not inferred — the edge is asserted directly by Semantic Scholar's own citation graph, not derived by this system (contrast with `candidate_gaps`' inferred patterns, which use `"medium"`/`"low"`).

## Backend: ingestion

**New module** `src/researchbridge/citations/fetch.py`:

```python
@dataclass
class CitationFetchResult:
    citing_paper_id: uuid.UUID
    cited_paper_id: uuid.UUID

def fetch_citation_edges(paper: Paper, api_key: str | None, ...) -> RawCitationsPayload:
    """One HTTP call to the S2 per-paper endpoint. Same throttle/retry shape as
    SemanticScholarConnector._fetch_page - reuses the same MIN_REQUEST_INTERVAL_SECONDS
    and _is_retryable from connectors/semantic_scholar.py rather than duplicating it."""
    ...

def resolve_and_persist(session: Session, paper: Paper, payload: RawCitationsPayload) -> CitationSummary:
    """Resolves citations/references paperIds against existing Paper rows, inserts
    PaperCitation rows for resolved pairs, skips the rest. Returns counts for the
    CLI summary."""
    ...
```

**New module** `src/researchbridge/citations/batch.py`, mirroring `gaps/batch.py::run_all`'s shape: iterate every `source="semantic_scholar"` paper (optionally skipping ones that already have outgoing edges unless `--force`), call `fetch_citation_edges` + `resolve_and_persist` per paper, aggregate a summary.

**New CLI** `src/researchbridge/citations/cli_fetch.py` (`rb-citations-fetch`), mirroring `gaps/cli_detect.py`:
- positional `source_id` (single paper) or `--all`
- `--force` (with `--all`, reprocess papers that already have outgoing edges)
- `--save` (persist; default is dry-run, printing what would be inserted) — same "dry run by default" discipline as `rb-gaps-detect`, since this writes real rows to a scientific-provenance table
- Registered in `pyproject.toml` as `rb-citations-fetch = "researchbridge.citations.cli_fetch:main"`

## Backend: citation graph endpoint

**New route** in `routes.py` (alongside the existing `/papers/{paper_id}/similar`):

```python
@router.get("/papers/{paper_id}/citations", response_model=CitationGraphOut)
def paper_citations(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> CitationGraphOut:
    ...
```

Loads the paper (404 if missing, same pattern as `get_paper`), then two queries against `paper_citations`: one where `cited_paper_id = paper_id` (papers citing this one) and one where `citing_paper_id = paper_id` (papers this one cites), joined to `Paper` for title. Assembles a `CitationGraphOut` with the target paper as the center node and each related paper as a satellite node, tagged with direction (`"cites"` or `"cited_by"`).

New schemas in `schemas.py`, mirroring `GraphNodeOut`/`GraphEdgeOut`/`SimilarityGraphOut`:

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

No embedding/similarity computation involved here — this is a direct table read, much cheaper than the similarity graph's build.

## Frontend

- `frontend/lib/api.ts`: add `citations: (id: string) => get<CitationGraphData>(\`/api/papers/${id}/citations\`)` next to the existing `similar()` entry.
- `frontend/components/CitationGraph.tsx`, structured like `SimilarityGraph.tsx`:
  - Same `ForceGraph2D` dynamic-import/THREE-stub machinery (extract the shared `loadForceGraph2D` helper into a small shared module if duplicating it verbatim starts to feel wrong — judgment call at implementation time, not a hard requirement here).
  - Center node = the current paper, visually distinct (same treatment `SimilarityGraph` gives the "input" node).
  - Edge color by direction: one color for "cites" (outgoing), another for "cited by" (incoming) — reuses existing color tokens, not the `--near`/`--far` distance ramp (there's no distance here, just direction).
  - Click a satellite node → side panel with title + a link to that paper's own detail page (`/papers/{id}`), same pattern `SimilarityGraph` uses for corpus papers.
  - Empty state: "No citation data yet for this paper" when both edge lists are empty — expected to be common until `rb-citations-fetch --all` has run and the corpus has grown enough for intra-corpus matches to exist.
- Wired into `app/papers/[id]/page.tsx` as a new section alongside the existing `similar` papers list and `ExtractedClaims`.

## Error handling

- Paper not found → existing 404 pattern from `get_paper`, unchanged.
- No citation edges either direction → not an error; empty graph, handled by the frontend's empty state.
- Citation fetch (`rb-citations-fetch`) hitting a 429/5xx from Semantic Scholar → same retry/backoff as the existing connector; a paper that still fails after retries is logged and skipped (summary counts it under "failed"), the batch continues rather than aborting.
- A citation/reference paperId that doesn't resolve to any in-corpus paper → not an error, just not persisted (counted separately in the CLI summary so the operator can see how sparse intra-corpus matching is).

## Testing

- Backend (`citations/fetch.py`, `citations/batch.py`): unit tests with a mocked Semantic Scholar response (via `responses`, same library the connector tests use) covering: both ends resolve (edge persisted), one end doesn't resolve (skipped), an edge that already exists (not duplicated, no error), and the retry/throttle behavior reusing the connector's own tested `_is_retryable`.
- CLI (`cli_fetch.py`): argument validation tests mirroring `gaps/cli_detect.py`'s `validate_args` tests (source_id vs `--all` vs `--force` combinations).
- Route test: `GET /api/papers/{id}/citations` returns 200 with the expected node/edge shape for a fixture with real `PaperCitation` rows, 404 for a nonexistent paper, and an empty-but-valid graph when no edges exist.
- Frontend: manually verified in-browser (dev server, click-through), consistent with `SimilarityGraph`'s lack of automated frontend tests.
