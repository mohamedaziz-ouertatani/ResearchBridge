# Interactive Claim & Evidence Graph — Design

Date: 2026-09-10

## Purpose

Give a reviewer a visual, explorable map of one assessment's reasoning: which
claims are backed by which verbatim evidence, whether that evidence supports
or contradicts the claim, and how the claims relate to the specific gap the
assessment is judging. A companion corpus-wide view shows which domain
categories carry the heaviest concentration of unaddressed gaps.

This is additive UI/API on top of existing data. No new tables, no new
columns, no migration.

## Scope decisions (resolved during brainstorming)

- **Confidence color**: no traffic-light palette. `ExtractedClaims.tsx`
  already states, deliberately, that color-coding confidence "would visually
  assert a trust signal the data doesn't support." The graph keeps that
  rule — confidence renders as plain-text tiers, with at most the existing
  ink-shade emphasis pattern used for `gap_status` badges. All new UI stays
  inside the existing `--ink`/`--ink-soft`/`--ink-faint`/`--rule`/`--rule-soft`
  token set, plus the reserved `--near`/`--far` ramp only if a proximity/
  distance value is ever shown (it isn't, in v1).
- **Graph library**: `@xyflow/react` + `dagre`, not the existing
  `react-force-graph-2d`. Force-directed layout is a physics simulation and
  is a poor fit for a stable DAG with custom node cards, a side inspector,
  and live filtering; React Flow is built for exactly that. This is a new
  frontend dependency, justified by the mismatch, not a default reuse of
  `react-force-graph-2d`/`d3-force-3d`.
- **`ADDRESSES_GAP` edges are derived, not stored.** There is no schema link
  between an arbitrary claim and a gap today. Within one assessment's scope,
  every claim being graphed **is** reasoning about that assessment's own gap
  (`ResearchAssessment.candidate_gap_id`), so the edge is synthesized: one
  `ADDRESSES_GAP` edge per assessment-scoped claim node → the gap node. No
  new column, no inference heuristic beyond the existing FK.
- **`contextualizes` relationship**: `ClaimEvidence.relationship` has three
  values (`supports`, `contradicts`, `contextualizes`), not two. All three
  render, as a third neutral edge style, rather than the spec's SUPPORTS/
  CONTRADICTS-only edge set silently dropping real data.
- **Gap density is corpus-wide**, not per-assessment — "density across
  domain categories" only means something in aggregate. `candidate_gaps` has
  no `domain`/`category` column; density is computed by joining
  `CandidateGap.seed_paper_id → Paper → PaperCategory.category` and grouping
  gap counts per category (a gap counts under every category its seed paper
  has).
- **No new charting library.** Nothing in `frontend/package.json` does
  heatmaps today (no recharts/visx). Rather than add a second new dependency
  alongside React Flow, the density overlay is a hand-rolled SVG/CSS grid
  using existing ink-shade intensity, and doubles as a highlight toggle on
  gap-type nodes/clusters inside the main graph.
- **No tab library exists in the frontend** (no shadcn/Radix `Tabs`
  anywhere). Rather than introduce one for a single feature, the graph is
  added as a plain button-toggled section (conditional render, existing
  Tailwind tokens) on the assessment page — consistent with how the page
  already just stacks sections (`AssessmentReport`, `SimilarityGraph`).
- **Route location**: no `app/services/` or `backend/app/api/routes/`
  layout exists in this repo (that's not how this codebase is structured —
  see `src/researchbridge/`). New routes go on the existing
  `assessment_routes.py` router (`/api/assessments` prefix), next to the
  existing `/graph` (similarity) endpoint, not a new standalone route file —
  there's no existing precedent for splitting routes per-feature. Business
  logic goes in `src/researchbridge/assessment/claim_graph.py` and
  `src/researchbridge/gaps/density.py`, mirroring the existing
  `assessment/graph.py` module (there is no `services/` layer in this
  codebase; domain packages hold logic, route handlers call them directly).

## Data model

**Claim graph, scoped to one assessment** (`assessment_id`):

- *Claim nodes*: `AnalysisClaim` rows where
  `source_table='research_assessments' AND source_id=assessment_id` (the
  Sec 16 claims already tied to this assessment), plus the single
  `AnalysisClaim` with `source_table='candidate_gaps'` whose `source_id` is
  `assessment.candidate_gap_id` (the claim describing the gap itself).
- *Evidence nodes*: every `Evidence` row reachable from those claim nodes via
  `ClaimEvidence`, plus every `Evidence` row reachable from the gap via
  `CandidateGapEvidence`.
- *Gap node*: the single `CandidateGap` at `assessment.candidate_gap_id`. If
  an assessment has no `candidate_gap_id`, the graph has no gap node and no
  `ADDRESSES_GAP` edges (claims still render with their evidence edges).
- *Edges*:
  - `SUPPORTS` / `CONTRADICTS` / `CONTEXTUALIZES` from
    `ClaimEvidence.relationship`, one edge per `ClaimEvidence` row, claim →
    evidence.
  - `ADDRESSES_GAP`: one synthesized edge per assessment-scoped claim node →
    gap node (see above). Not applied to the gap's own describing claim
    (that would be a self-referential loop in effect — the gap's claim
    doesn't "address" its own gap).

**Gap density** (corpus-wide, independent of any single assessment):
group all `candidate_gaps` by their seed paper's `PaperCategory.category`
values, counting `status != 'rejected'` gaps (rejected gaps aren't live
"unaddressed" gaps) per category. A gap with no `PaperCategory` row for its
seed paper falls into an `"uncategorized"` bucket rather than being dropped.

## Backend

### `src/researchbridge/assessment/claim_graph.py` (new)

Mirrors `assessment/graph.py`'s existing dataclass + builder-function shape:

```python
@dataclass
class ClaimGraphNode:
    id: str
    kind: Literal["claim", "evidence", "gap"]
    label: str
    # claim-only
    claim_type: str | None = None
    confidence: str | None = None
    status: str | None = None
    # evidence-only
    paper_id: uuid.UUID | None = None
    paper_title: str | None = None
    section: str | None = None
    # gap-only
    gap_status: str | None = None

@dataclass
class ClaimGraphEdge:
    source: str
    target: str
    relationship: Literal["supports", "contradicts", "contextualizes", "addresses_gap"]

@dataclass
class ClaimEvidenceGraph:
    nodes: list[ClaimGraphNode]
    edges: list[ClaimGraphEdge]

def build_claim_evidence_graph(session: Session, assessment: ResearchAssessment) -> ClaimEvidenceGraph: ...
```

Implementation queries, in order: assessment-scoped claims (as above) →
their `ClaimEvidence` rows (batched, same `.in_()` pattern as
`serializers.py::_evidence_by_gap`) → the target gap (if any) → its
`CandidateGapEvidence` rows → the gap's own describing claim + its
`ClaimEvidence` rows. Node ids are `str(uuid)`, matching the existing
similarity-graph convention.

### `src/researchbridge/gaps/density.py` (new)

```python
@dataclass
class GapDensityBucket:
    category: str
    gap_count: int

def compute_gap_density(session: Session) -> list[GapDensityBucket]: ...
```

One query: `CandidateGap` left-joined through `Paper` to `PaperCategory`,
filtered `CandidateGap.status != 'rejected'`, grouped by
`coalesce(PaperCategory.category, 'uncategorized')`, counting distinct gap
ids (a gap with two categories counts once per category, per the resolved
scope above — this is a `COUNT(DISTINCT gap.id)` per category group, not a
global distinct).

### Routes — added to `src/researchbridge/api/assessment_routes.py`

```python
@router.get("/{assessment_id}/claim-graph", response_model=ClaimEvidenceGraphOut)
def get_assessment_claim_graph(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> ClaimEvidenceGraphOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")
    graph = build_claim_evidence_graph(session, assessment)
    return ClaimEvidenceGraphOut(nodes=[...], edges=[...])

@router.get("/{assessment_id}/gap-density", response_model=GapDensityOut)
def get_gap_density(assessment_id: uuid.UUID, session: Session = Depends(get_session)) -> GapDensityOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")
    buckets = compute_gap_density(session)
    return GapDensityOut(buckets=[...])
```

`assessment_id` is required in the `gap-density` path (matching the spec's
literal endpoint shape and keeping both routes under the same resource) but
unused beyond the existence check — the response is corpus-wide, per the
resolved scope. No auth dependency, matching every other route in this
file (documented local-only tool convention).

### Schemas — added to `src/researchbridge/api/schemas.py`, next to `SimilarityGraphOut`

`ClaimGraphNodeOut`, `ClaimGraphEdgeOut`, `ClaimEvidenceGraphOut`,
`GapDensityBucketOut`, `GapDensityOut` — plain `BaseModel`s mirroring the
dataclass fields above, same convention as `GraphNodeOut`/`GraphEdgeOut`.

## Frontend

### Dependencies

```
npm install @xyflow/react dagre
npm install -D @types/dagre
```

### `frontend/components/graph/` (new)

- `ClaimEvidenceGraph.tsx` — fetches `/claim-graph`, runs dagre layout
  (top-to-bottom layered DAG: gap at bottom, evidence at top, claims in
  between — since evidence "supports up" into claims which "address down"
  into the gap), renders `<ReactFlow>` with custom node/edge types. Loaded
  via `next/dynamic` with SSR disabled, same pattern `SimilarityGraph.tsx`
  already uses for `react-force-graph-2d`.
- `ClaimNode.tsx` — claim text (truncated, full text in inspector), claim
  type, confidence as plain text, status.
- `EvidenceNode.tsx` — paper title, section, quoted excerpt (truncated).
- `GapNode.tsx` — observation (truncated), `gap_status` with the existing
  ink-shade badge treatment from `app/gaps/page.tsx`.
- `RelationshipEdge.tsx` — solid for `supports`, dashed for `contradicts`,
  dotted for `contextualizes`, distinct dash pattern for `addresses_gap`;
  all in ink/rule shades, not red/green.
- `InspectorDrawer.tsx` — side panel opened on node click: full claim text /
  full evidence passage + paper metadata (title, section, locator) / full
  gap observation + ratings, depending on node kind.
- `FilterToolbar.tsx` — toggle node-kind visibility (claim/evidence/gap) and
  confidence-level filter (high/medium/low), client-side filtering of the
  already-fetched graph (no refetch per filter change).
- `GapDensityOverlay.tsx` — fetches `/gap-density` once, toggle button that
  cross-references the current gap node's seed-paper categories against the
  density buckets and applies an ink-intensity highlight when the gap sits
  in a high-density category; when off, gap node renders plain.

### Integration — `frontend/app/assessments/[id]/page.tsx`

Add a button-toggled section below the existing stack:

```tsx
<div className="pt-12">
  <AssessmentReport assessment={assessment} onAssessmentUpdated={setAssessment} />
  <SimilarityGraph assessmentId={id} />
  <ClaimEvidenceGraph assessmentId={id} />
</div>
```

No new tab component — matches the page's existing plain vertical-stack
convention (confirmed: no tab/Tabs pattern exists anywhere in this
frontend). If `ClaimEvidenceGraph` later needs to be one of several tabs,
that's a separate follow-up, not part of this feature.

## Testing

- **Backend**: pytest tests for `build_claim_evidence_graph` (assessment
  with claims+evidence+gap; assessment with no gap; assessment with no
  claims) and `compute_gap_density` (multi-category gap, uncategorized gap,
  rejected gap excluded), following existing test-DB seeding patterns in
  `tests/`. Route-level tests for `/claim-graph` and `/gap-density`
  (200 + shape, 404 for missing assessment).
- **Frontend**: component tests for `ClaimNode`/`EvidenceNode`/`GapNode`
  rendering, `RelationshipEdge` style selection, and `FilterToolbar`
  filtering logic — consistent with the existing frontend test suite
  (104 tests as of 2026-09-03).

## Out of scope (v1)

- Editing claims/evidence/gap relationships from the graph view (read-only).
- Persisting `ADDRESSES_GAP` as a real schema relationship.
- Cross-assessment or corpus-wide claim graphs (this is per-assessment only;
  only gap-density is corpus-wide).
- A general tab component for the assessment page.
