# Assessment Similarity Graph — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint). This is a per-assessment visualization, not a corpus-wide knowledge graph — the blueprint is explicit that the product's core loop is ResearchInput → ResearchAssessment and that corpus-wide exploration is secondary supporting infrastructure, not something to prioritize as if it were the product. This feature reinforces the core loop: it visualizes one assessment's own retrieved-literature comparison, nothing more.

## Scope

In scope:
- One new endpoint, `GET /api/assessments/{id}/graph`, that returns a similarity graph for one assessment: the input as a center node, each retrieved paper as a node, weighted by embedding similarity — both input↔paper and paper↔paper edges.
- One new frontend component, `SimilarityGraph`, rendered on the existing assessment detail page (`app/assessments/[id]/page.tsx`), alongside `AssessmentReport`.
- Click-to-open side panel on a paper node: title, similarity to the input, and a claim-type summary (counts of problems/methods/limitations/etc. contributed by that paper) — all delivered in the graph payload itself, no second fetch.
- Computed on demand at request time, reusing the assessment's already-stored `retrieved_paper_ids` and the existing embedding infrastructure. No new migration, no persisted distances — works retroactively on every existing assessment.

Out of scope (deliberately deferred):
- Corpus-wide graph exploration (browsing the whole corpus as a graph, independent of any one assessment). Explicitly the secondary, deferred capability per the blueprint's product-loop framing.
- Persisting graph data anywhere. Stateless request/response, recomputed each time the page is viewed — same "recompute, don't persist derived state" pattern as `rerun_assessment`, which always rebuilds live rather than trusting stale stored state.
- An ANN index (ivfflat/hnsw) for the pairwise similarity computation. At Phase 1 corpus scale, a handful of retrieved papers (`top_k`, currently 10) means pairwise cosine distance is a trivial in-memory computation — same reasoning `embedding/search.py` already documents for deferring an index.
- Navigating away from the assessment page on node click. The side panel keeps the user in context; a link to the paper's own page can be added later if wanted, but isn't part of this feature.
- Frontend automated tests. No test infra exists on the frontend yet (confirmed via project memory); this stays manually verified in-browser like the rest of the frontend.

## Data flow

```
GET /api/assessments/{id}/graph
  -> load ResearchAssessment + its research_input (existing pattern, same as get_assessment)
  -> re-embed research_input.raw_text (or its representation, for uploaded docs -
     same query_text logic build_assessment already uses) with the request-scoped embedder
  -> load retrieved_paper_ids' Embedding vectors in ONE query (not N pgvector calls)
  -> compute input->paper cosine distances against the freshly embedded query vector
  -> compute paper->paper cosine distances pairwise, in Python/numpy, over the small
     retrieved set (bounded by top_k, ~10 papers => ~45 pairs at most)
  -> load each retrieved paper's non-stub claims (same query shape as
     build.py::_claims_for_paper), grouped into per-claim-type counts
  -> assemble GraphNode (input + one per paper) and GraphEdge (weight = 1 - distance) lists
  -> return as GraphOut
```

No new embedding pipeline, no new table, no new migration. The only new computation is an in-memory pairwise distance calculation over an already-small, bounded set of vectors already fetched from the existing `Embedding` table.

## Backend

**New module** `src/researchbridge/assessment/graph.py`:

```python
@dataclass
class GraphNode:
    id: str  # "input" or str(paper_id)
    type: Literal["input", "paper"]
    title: str
    similarity_to_input: float | None  # None for the input node itself
    claim_counts: dict[str, int]  # {} for the input node

@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float  # 1 - cosine_distance, in [0, 1]

@dataclass
class SimilarityGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

def build_similarity_graph(
    session: Session, assessment: ResearchAssessment, research_input: ResearchInput, embedder: Embedder,
) -> SimilarityGraph:
    ...
```

Reuses:
- `assessment/representation.py::build_research_representation` for the query text (same branch `build_assessment` uses for `input_type == "document"`).
- The same `Embedding` table `embedding/search.py` queries, but batched: one `SELECT` for all `retrieved_paper_ids`' vectors instead of per-pair pgvector round-trips. Pairwise cosine distance computed in Python (numpy) once vectors are in memory.
- `build.py::_claims_for_paper`'s query shape, but grouped into counts per `claim_type` rather than returned as raw claim text (the side panel shows "3 methods, 1 limitation," not full text — full text stays in the existing report/side-drawer patterns elsewhere in the app).

If `retrieved_paper_ids` is empty (no corpus match at assessment time), returns a `SimilarityGraph` with just the input node and no edges.

**New route** in `assessment_routes.py`:

```python
@router.get("/{assessment_id}/graph", response_model=GraphOut)
def get_assessment_graph(assessment_id: uuid.UUID, session=Depends(get_session), embedder=Depends(get_embedder)) -> GraphOut:
    ...
```

Follows the existing `get_assessment` pattern: 404 if the assessment doesn't exist, otherwise load its `research_input` and delegate to `build_similarity_graph`. New `GraphOut`/`GraphNodeOut`/`GraphEdgeOut` schemas in `schemas.py`, mirroring `ResearchAssessmentOut`'s style.

## Frontend

- Add `react-force-graph` as a new dependency (`frontend/package.json`).
- `frontend/lib/assessmentApi.ts`: add `graph(id: string): Promise<GraphData>` and the `GraphData`/`GraphNode`/`GraphEdge` types, following the existing `assessmentApi.get()` pattern.
- `frontend/components/SimilarityGraph.tsx`:
  - Renders the force graph via `react-force-graph` (2D, canvas-based).
  - Input node visually distinct (larger, different fill color from `--ink`/accent tokens already used elsewhere).
  - Paper nodes sized or colored by `similarity_to_input`; edge thickness by `weight`.
  - Click on a paper node opens a side panel (reusing existing Tailwind panel/border conventions from `AssessmentReport`) showing title, similarity as a percentage, and `claim_counts` as a small list.
  - Own independent loading/error state — a fetch failure here doesn't block `AssessmentReport` from rendering, same "each section owns its own fetch" pattern the page already follows for `assessment` vs. future sections.
  - Empty-edges case (only the input node returned): renders just that node with a small inline note ("no related papers found in the corpus") instead of an empty force-graph canvas.
- Wired into `app/assessments/[id]/page.tsx` as a new section alongside `AssessmentReport`.

## Error handling

- Assessment not found → existing 404 pattern, unchanged.
- Zero retrieved papers → not an error; empty-edges graph as described above.
- Embedder unavailable / embedding failure → propagates as a 500 like every other embedder-dependent route already does (no new handling needed, consistent with `create_assessment`).
- Frontend graph fetch failure → inline error scoped to the `SimilarityGraph` section only, doesn't affect the rest of the page.

## Testing

- Backend: unit test `build_similarity_graph` against a small fixture corpus (a handful of papers with real embeddings inserted via the test DB), asserting:
  - node count equals `1 + len(retrieved_paper_ids)`
  - edge weights are symmetric (paper A→B weight equals B→A)
  - the empty-retrieval case returns a single input node and no edges
  - claim_counts per node match the counts of that paper's own non-stub claims by type
- Route test: `GET /api/assessments/{id}/graph` returns 200 with the expected shape for an existing assessment fixture, and 404 for a nonexistent id.
- Frontend: manually verified in-browser (dev server, click-through), consistent with the rest of the frontend's lack of test infra.
