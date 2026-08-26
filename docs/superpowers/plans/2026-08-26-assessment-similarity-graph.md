# Assessment Similarity Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-assessment force-directed graph showing the input's similarity to the papers it was compared against, and those papers' similarity to each other, computed on demand from existing embedding data.

**Architecture:** A new backend module computes distances (input↔paper and paper↔paper) at request time from an assessment's already-stored `retrieved_paper_ids`, exposed via one new GET route. The frontend adds one new self-fetching component rendered alongside the existing assessment report, using `react-force-graph` for layout/rendering and the app's existing `--near`/`--far` cosine-distance color convention.

**Tech Stack:** FastAPI + SQLAlchemy (backend), pytest against a real Postgres test database (`tests/conftest.py`); Next.js App Router + TypeScript + Tailwind (frontend), `react-force-graph` (new dependency).

**Spec:** `docs/superpowers/specs/2026-08-26-assessment-similarity-graph-design.md`

## Global Constraints

- No new migration, no new table, no persisted distances — everything is computed at request time from the existing `Embedding` table (spec: "Data flow" / "Out of scope").
- Vectors are stored L2-normalized (see `Embedding.vector`'s docstring in `db/models.py`), so cosine similarity is a plain dot product and cosine distance is `1 - dot(a, b)` — same convention `qa/answer.py`'s `_dot` helper and `embedding/search.py`'s pgvector `cosine_distance` already use.
- A paper missing an embedding row for the current embedder's model is skipped from the graph entirely, never given a fabricated distance — "NULL is preferable to fabricated certainty" (blueprint Sec 22), same rule `build.py` follows everywhere else.
- Frontend must reuse the existing `--near`/`--far` teal-to-gray CSS variables (`frontend/app/globals.css`) for any cosine-distance encoding — that ramp is explicitly reserved for measured cosine distance and nothing else (see `AssessmentReport.tsx`'s header comment and `ProximityGauge.tsx`).
- Backend tests use the `FakeEmbedder` + `_hash_to_unit_vector` pattern already established in `tests/test_qa_answer.py` / `tests/test_assessment_build.py` (deterministic hash-based embedding, same text always embeds identically) rather than loading a real model.
- Run backend tests with `.venv/Scripts/python -m pytest <path> -v` (Windows venv, per existing plans in this repo).

---

## Task 1: `build_similarity_graph` domain function

**Files:**
- Create: `src/researchbridge/assessment/graph.py`
- Test: `tests/test_assessment_graph.py`

**Interfaces:**
- Consumes: `researchbridge.assessment.representation.build_research_representation(raw_text: str, embedder: Embedder) -> str`; `researchbridge.embedding.base.Embedder` (`.model_name: str`, `.embed_texts(texts: list[str]) -> list[list[float]]`); `researchbridge.embedding.pipeline.EMBEDDING_TYPE: str`; `researchbridge.db.models.{Embedding, Evidence, ExtractedClaim, Paper, ResearchAssessment, ResearchInput}`.
- Produces (used by Task 2): `GraphNode` (fields: `id: str`, `type: Literal["input", "paper"]`, `title: str`, `distance_to_input: float | None`, `claim_counts: dict[str, int]`), `GraphEdge` (fields: `source: str`, `target: str`, `distance: float`), `SimilarityGraph` (fields: `nodes: list[GraphNode]`, `edges: list[GraphEdge]`), and `build_similarity_graph(session: Session, assessment: ResearchAssessment, research_input: ResearchInput, embedder: Embedder) -> SimilarityGraph`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assessment_graph.py`:

```python
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from researchbridge.assessment.build import build_assessment
from researchbridge.assessment.graph import build_similarity_graph
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper, ResearchInput
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid") -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )


def _research_input(session, raw_text: str) -> ResearchInput:
    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=raw_text)
    session.add(ri)
    session.flush()
    return ri


def test_graph_has_input_node_plus_one_node_per_retrieved_paper(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _paper(session, embedder, "p2", "graph transformers for fraud detection variant")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    assert len(graph.nodes) == 1 + len(assessment.retrieved_paper_ids)
    assert graph.nodes[0].id == "input"
    assert graph.nodes[0].type == "input"
    assert graph.nodes[0].distance_to_input is None
    session.close()


def test_input_to_paper_distance_is_zero_for_exact_text_match(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "the exact same text")
    ri = _research_input(session, "the exact same text")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.distance_to_input == 0.0
    session.close()


def test_paper_to_paper_edges_are_symmetric(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _paper(session, embedder, "p2", "unrelated topic about weather prediction")
    ri = _research_input(session, "graph transformers for fraud")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=2)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_ids = [n.id for n in graph.nodes if n.type == "paper"]
    assert len(paper_ids) == 2
    edge = next(e for e in graph.edges if {e.source, e.target} == set(paper_ids))
    # only one edge per unordered pair - not a duplicate reverse edge
    assert sum(1 for e in graph.edges if {e.source, e.target} == set(paper_ids)) == 1
    assert 0.0 <= edge.distance <= 2.0
    session.close()


def test_claim_counts_grouped_by_claim_type(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "uses a graph transformer")
    _claim(session, paper, "method", "uses attention pooling")
    _claim(session, paper, "limitations", "evaluated offline only")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.claim_counts == {"method": 2, "limitations": 1}
    session.close()


def test_stub_claims_excluded_from_claim_counts(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "synthetic placeholder", extraction_method="stub")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.claim_counts == {}
    session.close()


def test_empty_retrieval_returns_only_input_node(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    ri = _research_input(session, "a topic with nothing in the corpus")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].type == "input"
    assert graph.edges == []
    session.close()


def test_paper_missing_embedding_for_current_model_is_skipped(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()
    assessment = build_assessment(session, ri.id, embedder)

    # simulate a model change after the assessment was built: no Embedding row
    # exists for "a-different-model", so the paper should be skipped, not error
    other_embedder = FakeEmbedder(model_name="a-different-model")
    graph = build_similarity_graph(session, assessment, ri, other_embedder)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].type == "input"
    assert graph.edges == []
    session.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_assessment_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.assessment.graph'`

- [ ] **Step 3: Write the implementation**

Create `src/researchbridge/assessment/graph.py`:

```python
"""Per-assessment similarity graph (blueprint Sec 2A, similarity-graph spec).

Computed on demand from an assessment's already-stored retrieved_paper_ids -
no new migration, no persisted distances, same "recompute rather than trust
stale stored state" pattern build.py::rerun_assessment already follows.

Vectors are L2-normalized (see Embedding.vector's docstring in db/models.py),
so cosine similarity is a plain dot product and cosine distance is
1 - dot(a, b) - same convention qa/answer.py's _dot helper uses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.representation import build_research_representation
from researchbridge.db.models import Embedding, Evidence, ExtractedClaim, Paper, ResearchAssessment, ResearchInput
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.pipeline import EMBEDDING_TYPE

_INPUT_TITLE_LENGTH = 80


@dataclass
class GraphNode:
    id: str
    type: Literal["input", "paper"]
    title: str
    distance_to_input: float | None
    claim_counts: dict[str, int]


@dataclass
class GraphEdge:
    source: str
    target: str
    distance: float


@dataclass
class SimilarityGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_similarity_graph(
    session: Session,
    assessment: ResearchAssessment,
    research_input: ResearchInput,
    embedder: Embedder,
) -> SimilarityGraph:
    input_node = GraphNode(
        id="input", type="input", title=_truncate(research_input.title or research_input.raw_text),
        distance_to_input=None, claim_counts={},
    )

    paper_ids = [uuid.UUID(pid) for pid in assessment.retrieved_paper_ids]
    if not paper_ids:
        return SimilarityGraph(nodes=[input_node], edges=[])

    query_text = (
        build_research_representation(research_input.raw_text, embedder)
        if research_input.input_type == "document"
        else research_input.raw_text
    )
    [query_vector] = embedder.embed_texts([query_text])

    # papers missing an embedding row for the current model are silently
    # excluded from this query's results - skipped below, not errored
    rows = session.execute(
        select(Paper.id, Paper.title, Embedding.vector)
        .join(Embedding, Embedding.paper_id == Paper.id)
        .where(
            Paper.id.in_(paper_ids),
            Embedding.embedding_type == EMBEDDING_TYPE,
            Embedding.model_name == embedder.model_name,
        )
    ).all()

    claim_counts_by_paper = _claim_counts(session, [row.id for row in rows])

    paper_nodes: list[GraphNode] = []
    vectors: dict[str, list[float]] = {}
    for row in rows:
        distance = 1.0 - _dot(query_vector, row.vector)
        node_id = str(row.id)
        vectors[node_id] = row.vector
        paper_nodes.append(
            GraphNode(
                id=node_id, type="paper", title=row.title, distance_to_input=distance,
                claim_counts=claim_counts_by_paper.get(row.id, {}),
            )
        )

    edges = [GraphEdge(source="input", target=node.id, distance=node.distance_to_input) for node in paper_nodes]
    node_ids = list(vectors.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            distance = 1.0 - _dot(vectors[node_ids[i]], vectors[node_ids[j]])
            edges.append(GraphEdge(source=node_ids[i], target=node_ids[j], distance=distance))

    return SimilarityGraph(nodes=[input_node, *paper_nodes], edges=edges)


def _claim_counts(session: Session, paper_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict[str, int]]:
    if not paper_ids:
        return {}
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.claim_type)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(ExtractedClaim.paper_id.in_(paper_ids), Evidence.extraction_method != "stub")
    ).all()
    counts: dict[uuid.UUID, dict[str, int]] = {}
    for paper_id, claim_type in rows:
        by_type = counts.setdefault(paper_id, {})
        by_type[claim_type] = by_type.get(claim_type, 0) + 1
    return counts


def _truncate(text: str) -> str:
    if len(text) > _INPUT_TITLE_LENGTH:
        return text[:_INPUT_TITLE_LENGTH] + "…"
    return text


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_assessment_graph.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/graph.py tests/test_assessment_graph.py
git commit -m "feat: compute per-assessment similarity graph on demand"
```

---

## Task 2: `GET /api/assessments/{id}/graph` route

**Files:**
- Modify: `src/researchbridge/api/schemas.py`
- Modify: `src/researchbridge/api/assessment_routes.py`
- Test: `tests/test_assessment_graph_api.py`

**Interfaces:**
- Consumes: Task 1's `build_similarity_graph(session, assessment, research_input, embedder) -> SimilarityGraph` (with `SimilarityGraph.nodes: list[GraphNode]`, `.edges: list[GraphEdge]`, `GraphNode.{id, type, title, distance_to_input, claim_counts}`, `GraphEdge.{source, target, distance}`); existing `get_session`/`get_embedder` deps from `researchbridge.api.deps`.
- Produces (used by Task 3): `GET /api/assessments/{assessment_id}/graph` returning JSON `{"nodes": [{"id": str, "type": "input"|"paper", "title": str, "distance_to_input": float|null, "claim_counts": {str: int}}], "edges": [{"source": str, "target": str, "distance": float}]}`; 404 with `{"detail": "No assessment with id ..."}` if the assessment doesn't exist.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_assessment_graph_api.py`:

```python
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def client(session_factory, embedder):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def _add_paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def test_graph_returns_input_and_paper_nodes(client, session, embedder) -> None:
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    created = client.post(
        "/api/assessments", json={"raw_text": "graph transformers for fraud detection"}
    ).json()

    response = client.get(f"/api/assessments/{created['id']}/graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 2
    input_node = next(n for n in body["nodes"] if n["type"] == "input")
    paper_node = next(n for n in body["nodes"] if n["type"] == "paper")
    assert input_node["distance_to_input"] is None
    assert paper_node["distance_to_input"] == pytest.approx(0.0, abs=1e-6)
    assert len(body["edges"]) == 1
    assert body["edges"][0]["source"] == "input"


def test_graph_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/graph")

    assert response.status_code == 404


def test_graph_returns_only_input_node_for_empty_corpus(client, session) -> None:
    created = client.post("/api/assessments", json={"raw_text": "nothing in the corpus"}).json()

    response = client.get(f"/api/assessments/{created['id']}/graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 1
    assert body["edges"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_assessment_graph_api.py -v`
Expected: FAIL with 404 "Not Found" (route doesn't exist yet) on the first two tests

- [ ] **Step 3: Write the implementation**

In `src/researchbridge/api/schemas.py`, add (near `ResearchAssessmentOut`):

```python
class GraphNodeOut(BaseModel):
    id: str
    type: Literal["input", "paper"]
    title: str
    distance_to_input: float | None
    claim_counts: dict[str, int]


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    distance: float


class SimilarityGraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
```

(Add `from typing import Literal` to the top of `schemas.py` if not already imported — check first; if `Literal` is already imported elsewhere in the file, reuse that import line.)

In `src/researchbridge/api/assessment_routes.py`:

Add to the imports at the top:
```python
from researchbridge.api.schemas import (
    ...,
    SimilarityGraphOut,
    GraphNodeOut,
    GraphEdgeOut,
)
from researchbridge.assessment.graph import build_similarity_graph
```

Add the route, next to `get_assessment`:

```python
@router.get("/{assessment_id}/graph", response_model=SimilarityGraphOut)
def get_assessment_graph(
    assessment_id: uuid.UUID,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> SimilarityGraphOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    research_input = session.get(ResearchInput, assessment.research_input_id)
    graph = build_similarity_graph(session, assessment, research_input, embedder)

    return SimilarityGraphOut(
        nodes=[
            GraphNodeOut(
                id=node.id, type=node.type, title=node.title,
                distance_to_input=node.distance_to_input, claim_counts=node.claim_counts,
            )
            for node in graph.nodes
        ],
        edges=[GraphEdgeOut(source=edge.source, target=edge.target, distance=edge.distance) for edge in graph.edges],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_assessment_graph_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full backend test suite to check for regressions**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: PASS (all tests, including the pre-existing suite)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/assessment_routes.py tests/test_assessment_graph_api.py
git commit -m "feat: expose GET /api/assessments/{id}/graph"
```

---

## Task 3: Frontend API client for the graph endpoint

**Files:**
- Modify: `frontend/package.json` (add `react-force-graph` dependency)
- Modify: `frontend/lib/assessmentApi.ts`

**Interfaces:**
- Consumes: `GET /api/assessments/{id}/graph` from Task 2, returning `{nodes: [{id, type, title, distance_to_input, claim_counts}], edges: [{source, target, distance}]}`.
- Produces (used by Task 4): `assessmentApi.graph(id: string): Promise<GraphData>`, and exported types `GraphNode`, `GraphEdge`, `GraphData` from `frontend/lib/assessmentApi.ts`.

- [ ] **Step 1: Install the dependency**

Run: `cd frontend && npm install react-force-graph`
Expected: `frontend/package.json` and `frontend/package-lock.json` updated with `react-force-graph` under `dependencies`.

- [ ] **Step 2: Add the types and API method**

In `frontend/lib/assessmentApi.ts`, add near the other type definitions (after `AssessmentSummaryPage`):

```typescript
export type GraphNode = {
  id: string;
  type: "input" | "paper";
  title: string;
  distance_to_input: number | null;
  claim_counts: Record<string, number>;
};

export type GraphEdge = {
  source: string;
  target: string;
  distance: number;
};

export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};
```

Add to the `assessmentApi` object (after `history`):

```typescript
  graph: (id: string) => request<GraphData>(`/api/assessments/${id}/graph`),
```

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new type errors (pre-existing errors, if any, are unrelated — compare against a run on `master` before this change if unsure)

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/lib/assessmentApi.ts
git commit -m "feat: add graph endpoint to the assessment API client"
```

---

## Task 4: `SimilarityGraph` component

**Files:**
- Create: `frontend/components/SimilarityGraph.tsx`
- Modify: `frontend/app/assessments/[id]/page.tsx`

**Interfaces:**
- Consumes: Task 3's `assessmentApi.graph(id)` and its `GraphData`/`GraphNode`/`GraphEdge` types; the app's `--near`/`--far`/`--panel`/`--rule`/`--ink-soft` CSS variables (`frontend/app/globals.css`); the `eyebrow` utility class already used throughout the frontend.
- Produces: `<SimilarityGraph assessmentId={string} />`, a self-fetching component with no other props, following the same pattern as `AssessmentHistory` in `components/AssessmentReport.tsx`.

- [ ] **Step 1: Write the component**

Create `frontend/components/SimilarityGraph.tsx`:

```tsx
"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { assessmentApi, type GraphData, type GraphNode } from "@/lib/assessmentApi";

// react-force-graph draws to a <canvas> and reads `window` at import time -
// it cannot run during server-side rendering, same reason ProximityGauge's
// sibling gauge components stay client components.
const ForceGraph2D = dynamic(() => import("react-force-graph").then((mod) => mod.ForceGraph2D), { ssr: false });

/** Same "teal near, washed-out far" convention as ProximityGauge - this ramp
 * is reserved for measured cosine distance and nothing else (see globals.css). */
function distanceColor(distance: number): string {
  const position = Math.min(Math.max(distance, 0), 1) * 100;
  return `color-mix(in oklab, var(--near), var(--far) ${position}%)`;
}

export function SimilarityGraph({ assessmentId }: { assessmentId: string }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);

  useEffect(() => {
    assessmentApi
      .graph(assessmentId)
      .then(setData)
      .catch(() => setError("The similarity graph isn't available."));
  }, [assessmentId]);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.edges.map((e) => ({ source: e.source, target: e.target, distance: e.distance })),
    };
  }, [data]);

  if (error) {
    return <p className="py-6 text-[0.8125rem] text-[var(--ink-soft)]">{error}</p>;
  }

  if (!data) {
    return <p className="eyebrow py-6">loading graph…</p>;
  }

  const hasPapers = data.nodes.some((n) => n.type === "paper");

  return (
    <section className="border-t border-[var(--rule)] py-8">
      <span className="eyebrow">similarity graph</span>

      {!hasPapers ? (
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">
          No related papers found in the corpus.
        </p>
      ) : (
        <div className="mt-4 flex flex-col gap-4 sm:flex-row">
          <div className="h-[420px] flex-1 border border-[var(--rule)]">
            <ForceGraph2D
              graphData={graphData}
              nodeId="id"
              nodeLabel={(node: GraphNode) => node.title}
              nodeColor={(node: GraphNode) =>
                node.type === "input" ? "var(--ink)" : distanceColor(node.distance_to_input ?? 1)
              }
              nodeVal={(node: GraphNode) => (node.type === "input" ? 8 : 4)}
              linkColor={(link: { distance: number }) => distanceColor(link.distance)}
              linkWidth={(link: { distance: number }) => Math.max(0.5, 3 * (1 - link.distance))}
              onNodeClick={(node: GraphNode) => setSelected(node.type === "paper" ? node : null)}
            />
          </div>

          {selected && (
            <aside className="w-full shrink-0 border border-[var(--rule)] bg-[var(--panel)] p-4 sm:w-64">
              <p className="text-[0.9375rem] leading-[1.5]">{selected.title}</p>
              <p className="eyebrow mt-3">
                distance to input: {selected.distance_to_input?.toFixed(3) ?? "—"}
              </p>
              {Object.keys(selected.claim_counts).length === 0 ? (
                <p className="mt-2 text-[0.8125rem] text-[var(--ink-soft)]">No extracted claims.</p>
              ) : (
                <ul className="mt-2 space-y-1">
                  {Object.entries(selected.claim_counts).map(([claimType, count]) => (
                    <li key={claimType} className="text-[0.8125rem] text-[var(--ink-soft)]">
                      {count} × {claimType.replace(/_/g, " ")}
                    </li>
                  ))}
                </ul>
              )}
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Wire it into the assessment page**

In `frontend/app/assessments/[id]/page.tsx`, add the import:

```typescript
import { SimilarityGraph } from "@/components/SimilarityGraph";
```

Change the render block from:

```tsx
      {assessment && (
        <div className="pt-12">
          <AssessmentReport assessment={assessment} />
        </div>
      )}
```

to:

```tsx
      {assessment && (
        <div className="pt-12">
          <AssessmentReport assessment={assessment} />
          <SimilarityGraph assessmentId={id} />
        </div>
      )}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new type errors

- [ ] **Step 4: Manually verify in the browser**

Run: `cd frontend && npm run dev`, then open an existing assessment's detail page (`/assessments/{id}`) for an assessment with at least one retrieved paper.

Check:
- The graph section renders below the existing report, with an input node and one node per retrieved paper.
- Clicking a paper node opens the side panel showing its title, distance, and claim counts.
- Node/edge colors follow the teal-near / gray-far convention (compare visually against the existing `ProximityGauge` on another page, e.g. `/ask` or `/papers`, for the same color scale).
- An assessment with zero retrieved papers shows the "No related papers found" message instead of an empty canvas.
- Stop the dev server when done.

- [ ] **Step 5: Commit**

```bash
git add frontend/components/SimilarityGraph.tsx frontend/app/assessments/[id]/page.tsx
git commit -m "feat: render the assessment similarity graph on the assessment page"
```
