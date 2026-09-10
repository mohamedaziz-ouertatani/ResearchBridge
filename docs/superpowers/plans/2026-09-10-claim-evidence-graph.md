# Interactive Claim & Evidence Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a reviewer an interactive DAG of one assessment's claims, verbatim evidence, and target gap — plus a corpus-wide gap-density-by-category view — as a new section on the assessment page.

**Architecture:** Two new read-only backend endpoints (`/claim-graph`, `/gap-density`) on the existing `assessment_routes.py` router, backed by two new query modules (`assessment/claim_graph.py`, `gaps/density.py`) that mirror the existing `assessment/graph.py` dataclass-builder pattern. A new `@xyflow/react` + `@dagrejs/dagre` frontend subsystem under `frontend/components/graph/` renders the graph, replacing force-directed layout (used by the two existing graphs) with a stable layered DAG suited to custom node cards and a side inspector.

**Tech Stack:** FastAPI + SQLAlchemy 2.0 (backend, existing), Next.js + React + `@xyflow/react` + `@dagrejs/dagre` (frontend, two new deps), vitest + `@testing-library/react` (frontend tests, existing), pytest (backend tests, existing).

**Spec:** [docs/superpowers/specs/2026-09-10-claim-evidence-graph-design.md](../specs/2026-09-10-claim-evidence-graph-design.md)

## Global Constraints

- No new database tables or columns — everything is computed from existing tables (`analysis_claims`, `claim_evidence`, `evidence`, `candidate_gaps`, `candidate_gap_evidence`, `paper_categories`).
- No color-coded confidence/trust signal anywhere in the new UI. Stay inside the existing `--ink` / `--ink-soft` / `--ink-faint` / `--rule` / `--rule-soft` token set (see `frontend/app/globals.css`). Never introduce emerald/amber/rose or red/green.
- No new charting/graph library beyond `@xyflow/react` + `@dagrejs/dagre`. No shadcn/Radix — the codebase has no tab/dialog component library; new UI uses plain elements + existing Tailwind tokens.
- Every new route follows existing convention: no auth dependency, `session.get(...)` + `HTTPException(404, ...)` for a missing assessment, `response_model=<PydanticOut>`.
- `ClaimEvidence.relationship` has three real values — `supports`, `contradicts`, `contextualizes` — all three must render, not just SUPPORTS/CONTRADICTS.
- `ADDRESSES_GAP` edges are synthesized (claim → assessment's own gap), never read from a stored column — there is no such column.

---

## Task 1: Backend claim/evidence graph builder

**Files:**
- Create: `src/researchbridge/assessment/claim_graph.py`
- Test: `tests/test_claim_graph.py`

**Interfaces:**
- Produces: `ClaimGraphNode` (dataclass: `id: str`, `kind: Literal["claim","evidence","gap"]`, `label: str`, `claim_type: str | None = None`, `confidence: str | None = None`, `status: str | None = None`, `paper_id: str | None = None`, `paper_title: str | None = None`, `section: str | None = None`, `gap_status: str | None = None`, `categories: list[str] = field(default_factory=list)`), `ClaimGraphEdge` (dataclass: `source: str`, `target: str`, `relationship: Literal["supports","contradicts","contextualizes","addresses_gap"]`), `ClaimEvidenceGraph` (dataclass: `nodes: list[ClaimGraphNode]`, `edges: list[ClaimGraphEdge]`), and `build_claim_evidence_graph(session: Session, assessment: ResearchAssessment) -> ClaimEvidenceGraph`. Task 3 imports all of these.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for build_claim_evidence_graph (src/researchbridge/assessment/claim_graph.py)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from researchbridge.assessment.claim_graph import build_claim_evidence_graph
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    ClaimEvidence,
    Evidence,
    Paper,
    PaperCategory,
    ResearchAssessment,
    ResearchInput,
)


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    # analysis_claims/claim_evidence/candidate_gaps/candidate_gap_evidence and
    # research_assessments/research_inputs aren't in conftest's TRUNCATE list
    s.execute(
        text(
            "TRUNCATE TABLE claim_evidence, analysis_claims, candidate_gap_evidence, candidate_gaps, "
            "research_assessments, research_inputs CASCADE"
        )
    )
    s.commit()
    s.close()


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_evidence(session, paper: Paper, text_: str) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="results", section="results", text=text_,
        extraction_method="hybrid", model_version="v1", confidence="high",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _add_assessment(session, candidate_gap_id: uuid.UUID | None = None) -> ResearchAssessment:
    research_input = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(id=uuid.uuid4(), research_input_id=research_input.id, candidate_gap_id=candidate_gap_id)
    session.add(assessment)
    session.flush()
    return assessment


def _add_assessment_claim(session, assessment: ResearchAssessment, evidence: Evidence, relationship: str = "supports") -> AnalysisClaim:
    claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="fact", claim_text="the comparison summary claim", confidence="high",
        status="pending", source_table="research_assessments", source_id=assessment.id,
    )
    session.add(claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relationship=relationship))
    session.commit()
    return claim


def _add_gap_with_claim(session, seed_paper: Paper, evidence: Evidence) -> tuple[CandidateGap, AnalysisClaim]:
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed_paper.id, observation="a recurring unaddressed limitation",
        gap_type="inference", status="pending", contributing_paper_count=3, similarity_threshold=0.35,
        detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    gap_claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="inference", claim_text=gap.observation, confidence="medium",
        status=gap.status, source_table="candidate_gaps", source_id=gap.id,
    )
    session.add(gap_claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=gap_claim.id, evidence_id=evidence.id, relationship="supports"))
    session.commit()
    return gap, gap_claim


def test_returns_assessment_claim_node_with_its_evidence_node_and_edge(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    assessment = _add_assessment(session)
    claim = _add_assessment_claim(session, assessment, evidence)

    graph = build_claim_evidence_graph(session, assessment)

    claim_nodes = [n for n in graph.nodes if n.kind == "claim"]
    evidence_nodes = [n for n in graph.nodes if n.kind == "evidence"]
    assert [n.id for n in claim_nodes] == [str(claim.id)]
    assert claim_nodes[0].claim_type == "fact"
    assert claim_nodes[0].confidence == "high"
    assert [n.id for n in evidence_nodes] == [str(evidence.id)]
    assert evidence_nodes[0].label == "quoted passage"
    assert evidence_nodes[0].paper_title == "a paper"
    assert len(graph.edges) == 1
    assert graph.edges[0].source == str(claim.id)
    assert graph.edges[0].target == str(evidence.id)
    assert graph.edges[0].relationship == "supports"


def test_preserves_contradicts_and_contextualizes_relationships(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "a conflicting passage")
    assessment = _add_assessment(session)
    _add_assessment_claim(session, assessment, evidence, relationship="contradicts")

    graph = build_claim_evidence_graph(session, assessment)

    assert graph.edges[0].relationship == "contradicts"


def test_includes_gap_node_with_categories_and_addresses_gap_edges(session) -> None:
    seed_paper = _add_paper(session, "seed paper")
    session.add(PaperCategory(paper_id=seed_paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    gap_evidence = _add_evidence(session, seed_paper, "gap-supporting passage")
    gap, gap_claim = _add_gap_with_claim(session, seed_paper, gap_evidence)

    other_paper = _add_paper(session, "other paper")
    assessment_evidence = _add_evidence(session, other_paper, "assessment evidence passage")
    assessment = _add_assessment(session, candidate_gap_id=gap.id)
    assessment_claim = _add_assessment_claim(session, assessment, assessment_evidence)

    graph = build_claim_evidence_graph(session, assessment)

    gap_nodes = [n for n in graph.nodes if n.kind == "gap"]
    assert len(gap_nodes) == 1
    assert gap_nodes[0].id == str(gap.id)
    assert gap_nodes[0].gap_status == gap.gap_status
    assert gap_nodes[0].categories == ["Machine Learning"]

    claim_nodes = {n.id for n in graph.nodes if n.kind == "claim"}
    assert claim_nodes == {str(assessment_claim.id), str(gap_claim.id)}

    addresses_edges = [e for e in graph.edges if e.relationship == "addresses_gap"]
    assert [e.source for e in addresses_edges] == [str(assessment_claim.id)]
    assert addresses_edges[0].target == str(gap.id)


def test_no_gap_node_when_assessment_has_no_candidate_gap(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    assessment = _add_assessment(session)
    _add_assessment_claim(session, assessment, evidence)

    graph = build_claim_evidence_graph(session, assessment)

    assert not any(n.kind == "gap" for n in graph.nodes)
    assert not any(e.relationship == "addresses_gap" for e in graph.edges)


def test_empty_graph_for_assessment_with_no_claims_and_no_gap(session) -> None:
    assessment = _add_assessment(session)

    graph = build_claim_evidence_graph(session, assessment)

    assert graph.nodes == []
    assert graph.edges == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_claim_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.assessment.claim_graph'`

- [ ] **Step 3: Write the implementation**

```python
"""Per-assessment claim/evidence/gap graph (Interactive Claim & Evidence Graph spec).

Computed on demand, same "recompute rather than persist a derived view"
pattern as assessment/graph.py. ADDRESSES_GAP edges are synthesized, not
read from a column - within one assessment's scope, every assessment-derived
claim is by definition reasoning about that assessment's own
candidate_gap_id, so one such edge is added per assessment claim -> gap. The
gap's own describing claim (source_table="candidate_gaps", see
gaps/claims.py) is included as an ordinary claim node with its own evidence
via claim_evidence - CandidateGapEvidence is redundant for this graph
because gaps/claims.py::save_claim_for_gap already mirrors every
CandidateGapEvidence link into a claim_evidence "supports" row when it
creates that claim.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import AnalysisClaim, CandidateGap, ClaimEvidence, Evidence, Paper, PaperCategory, ResearchAssessment


@dataclass
class ClaimGraphNode:
    id: str
    kind: Literal["claim", "evidence", "gap"]
    label: str
    claim_type: str | None = None
    confidence: str | None = None
    status: str | None = None
    paper_id: str | None = None
    paper_title: str | None = None
    section: str | None = None
    gap_status: str | None = None
    categories: list[str] = field(default_factory=list)


@dataclass
class ClaimGraphEdge:
    source: str
    target: str
    relationship: Literal["supports", "contradicts", "contextualizes", "addresses_gap"]


@dataclass
class ClaimEvidenceGraph:
    nodes: list[ClaimGraphNode]
    edges: list[ClaimGraphEdge]


def build_claim_evidence_graph(session: Session, assessment: ResearchAssessment) -> ClaimEvidenceGraph:
    assessment_claims = list(
        session.execute(
            select(AnalysisClaim).where(
                AnalysisClaim.source_table == "research_assessments",
                AnalysisClaim.source_id == assessment.id,
            )
        ).scalars()
    )
    assessment_claim_ids = {c.id for c in assessment_claims}

    gap: CandidateGap | None = None
    gap_claim: AnalysisClaim | None = None
    if assessment.candidate_gap_id is not None:
        gap = session.get(CandidateGap, assessment.candidate_gap_id)
        if gap is not None:
            gap_claim = session.execute(
                select(AnalysisClaim).where(
                    AnalysisClaim.source_table == "candidate_gaps", AnalysisClaim.source_id == gap.id
                )
            ).scalar_one_or_none()

    all_claims = list(assessment_claims)
    if gap_claim is not None:
        all_claims.append(gap_claim)

    claim_ids = [c.id for c in all_claims]
    evidence_rows = (
        session.execute(
            select(ClaimEvidence, Evidence, Paper.title)
            .join(Evidence, Evidence.id == ClaimEvidence.evidence_id)
            .join(Paper, Paper.id == Evidence.paper_id)
            .where(ClaimEvidence.claim_id.in_(claim_ids))
        ).all()
        if claim_ids
        else []
    )

    evidence_nodes: dict[uuid.UUID, ClaimGraphNode] = {}
    edges: list[ClaimGraphEdge] = []
    for link, evidence, paper_title in evidence_rows:
        if evidence.id not in evidence_nodes:
            evidence_nodes[evidence.id] = ClaimGraphNode(
                id=str(evidence.id), kind="evidence", label=evidence.text,
                paper_id=str(evidence.paper_id), paper_title=paper_title, section=evidence.section,
            )
        edges.append(
            ClaimGraphEdge(source=str(link.claim_id), target=str(evidence.id), relationship=link.relationship)
        )

    claim_nodes = [
        ClaimGraphNode(
            id=str(c.id), kind="claim", label=c.claim_text,
            claim_type=c.claim_type, confidence=c.confidence, status=c.status,
        )
        for c in all_claims
    ]

    nodes: list[ClaimGraphNode] = [*claim_nodes, *evidence_nodes.values()]

    if gap is not None:
        categories = [
            row
            for row in session.execute(
                select(PaperCategory.category).where(PaperCategory.paper_id == gap.seed_paper_id)
            ).scalars()
        ]
        nodes.append(
            ClaimGraphNode(
                id=str(gap.id), kind="gap", label=gap.observation, gap_status=gap.gap_status, categories=categories,
            )
        )
        for claim_id in assessment_claim_ids:
            edges.append(ClaimGraphEdge(source=str(claim_id), target=str(gap.id), relationship="addresses_gap"))

    return ClaimEvidenceGraph(nodes=nodes, edges=edges)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_claim_graph.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/claim_graph.py tests/test_claim_graph.py
git commit -m "$(cat <<'EOF'
feat(assessment): add claim/evidence/gap graph builder

Computes a per-assessment DAG of claims, their verbatim evidence, and
the assessment's target gap, mirroring assessment/graph.py's dataclass
pattern. ADDRESSES_GAP edges are synthesized from
ResearchAssessment.candidate_gap_id - no such edge exists in the
schema.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Backend gap density module

**Files:**
- Create: `src/researchbridge/gaps/density.py`
- Test: `tests/test_gap_density.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `GapDensityBucket` (dataclass: `category: str`, `gap_count: int`) and `compute_gap_density(session: Session) -> list[GapDensityBucket]`. Task 3 imports both.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for compute_gap_density (src/researchbridge/gaps/density.py)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from researchbridge.db.models import CandidateGap, Paper, PaperCategory
from researchbridge.gaps.density import UNCATEGORIZED, compute_gap_density


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.execute(text("TRUNCATE TABLE candidate_gap_evidence, candidate_gaps CASCADE"))
    s.commit()
    s.close()


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_gap(session, seed_paper: Paper, status: str = "pending") -> CandidateGap:
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed_paper.id, observation="an observation", gap_type="inference",
        status=status, contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    return gap


def test_groups_gap_counts_by_seed_papers_category(session) -> None:
    ml_paper = _add_paper(session, "ml paper")
    session.add(PaperCategory(paper_id=ml_paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    robotics_paper = _add_paper(session, "robotics paper")
    session.add(PaperCategory(paper_id=robotics_paper.id, category="Robotics", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, ml_paper)
    _add_gap(session, ml_paper)
    _add_gap(session, robotics_paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {"Machine Learning": 2, "Robotics": 1}


def test_gap_with_no_paper_category_falls_into_uncategorized_bucket(session) -> None:
    paper = _add_paper(session, "an uncategorized paper")
    _add_gap(session, paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {UNCATEGORIZED: 1}


def test_rejected_gaps_are_excluded(session) -> None:
    paper = _add_paper(session, "a paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, paper, status="rejected")
    session.commit()

    buckets = compute_gap_density(session)

    assert buckets == []


def test_gap_counted_once_per_category_even_with_multiple_seed_categories(session) -> None:
    paper = _add_paper(session, "a multi-category paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.add(PaperCategory(paper_id=paper.id, category="Robotics", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {"Machine Learning": 1, "Robotics": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gap_density.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.gaps.density'`

- [ ] **Step 3: Write the implementation**

```python
"""Corpus-wide gap density by seed-paper domain category.

candidate_gaps has no domain/category column of its own - density is
computed by joining through seed_paper_id -> papers -> paper_categories.
A gap counts once per category its seed paper has (a multi-category paper's
gap appears in every one of those buckets), and once per category, not once
per contributing paper - only the seed paper's categories are used, since
that's the only 1:1 join available (a gap can have many contributing
papers, each with its own categories).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, Paper, PaperCategory

UNCATEGORIZED = "uncategorized"


@dataclass
class GapDensityBucket:
    category: str
    gap_count: int


def compute_gap_density(session: Session) -> list[GapDensityBucket]:
    rows = session.execute(
        select(PaperCategory.category, CandidateGap.id)
        .select_from(CandidateGap)
        .join(Paper, Paper.id == CandidateGap.seed_paper_id)
        .join(PaperCategory, PaperCategory.paper_id == Paper.id, isouter=True)
        .where(CandidateGap.status != "rejected")
    ).all()

    gap_ids_by_category: dict[str, set] = {}
    for category, gap_id in rows:
        bucket = category or UNCATEGORIZED
        gap_ids_by_category.setdefault(bucket, set()).add(gap_id)

    return sorted(
        (GapDensityBucket(category=category, gap_count=len(gap_ids)) for category, gap_ids in gap_ids_by_category.items()),
        key=lambda b: (-b.gap_count, b.category),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gap_density.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/gaps/density.py tests/test_gap_density.py
git commit -m "$(cat <<'EOF'
feat(gaps): add corpus-wide gap density by category

candidate_gaps has no domain/category column, so density is computed
by joining seed_paper_id -> papers -> paper_categories and counting
non-rejected gaps per category.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: API schemas and routes

**Files:**
- Modify: `src/researchbridge/api/schemas.py` (append near `SimilarityGraphOut`, after line 483)
- Modify: `src/researchbridge/api/assessment_routes.py` (add imports; add two routes after the existing `/graph` route, i.e. after line 317)
- Test: `tests/test_claim_evidence_graph_api.py`

**Interfaces:**
- Consumes: `ClaimGraphNode`, `ClaimGraphEdge`, `ClaimEvidenceGraph`, `build_claim_evidence_graph` from Task 1 (`researchbridge.assessment.claim_graph`); `GapDensityBucket`, `compute_gap_density` from Task 2 (`researchbridge.gaps.density`).
- Produces: `GET /api/assessments/{assessment_id}/claim-graph` → `ClaimEvidenceGraphOut`, `GET /api/assessments/{assessment_id}/gap-density` → `GapDensityOut`. Task 4 (frontend) consumes these two response shapes verbatim.

- [ ] **Step 1: Write the failing tests**

```python
"""Route tests for /claim-graph and /gap-density (see test_claim_graph.py /
test_gap_density.py for the underlying builder unit tests)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    ClaimEvidence,
    Evidence,
    Paper,
    PaperCategory,
    ResearchAssessment,
    ResearchInput,
)


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.execute(
        text(
            "TRUNCATE TABLE claim_evidence, analysis_claims, candidate_gap_evidence, candidate_gaps, "
            "research_assessments, research_inputs CASCADE"
        )
    )
    s.commit()
    s.close()


@pytest.fixture()
def client(session_factory):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    with TestClient(app) as c:
        yield c


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_evidence(session, paper: Paper, text_: str) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="results", section="results", text=text_,
        extraction_method="hybrid", model_version="v1", confidence="high",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _add_assessment_with_claim(session, candidate_gap_id=None):
    research_input = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(id=uuid.uuid4(), research_input_id=research_input.id, candidate_gap_id=candidate_gap_id)
    session.add(assessment)
    session.flush()
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="fact", claim_text="a claim", confidence="high", status="pending",
        source_table="research_assessments", source_id=assessment.id,
    )
    session.add(claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relationship="supports"))
    session.commit()
    return assessment


def test_claim_graph_returns_claims_and_evidence(client, session) -> None:
    assessment = _add_assessment_with_claim(session)

    response = client.get(f"/api/assessments/{assessment.id}/claim-graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) == 2
    assert {n["kind"] for n in body["nodes"]} == {"claim", "evidence"}
    assert len(body["edges"]) == 1
    assert body["edges"][0]["relationship"] == "supports"


def test_claim_graph_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/claim-graph")

    assert response.status_code == 404


def test_gap_density_returns_buckets(client, session) -> None:
    paper = _add_paper(session, "a paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=paper.id, observation="an observation", gap_type="inference",
        status="pending", contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.commit()
    assessment = _add_assessment_with_claim(session)

    response = client.get(f"/api/assessments/{assessment.id}/gap-density")

    assert response.status_code == 200
    body = response.json()
    assert body["buckets"] == [{"category": "Machine Learning", "gap_count": 1}]


def test_gap_density_returns_404_for_missing_assessment(client) -> None:
    response = client.get(f"/api/assessments/{uuid.uuid4()}/gap-density")

    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_claim_evidence_graph_api.py -v`
Expected: FAIL with 404-vs-200 mismatches / `AttributeError` — the routes don't exist yet (FastAPI returns 404 for an unmatched path too, but the "returns claims and evidence" test's `response.json()` shape assertions will fail since the path doesn't route to a handler that returns that body — this is expected)

- [ ] **Step 3: Add schemas**

In `src/researchbridge/api/schemas.py`, add immediately after `SimilarityGraphOut` (after line 483):

```python
class ClaimGraphNodeOut(BaseModel):
    id: str
    kind: Literal["claim", "evidence", "gap"]
    label: str
    claim_type: str | None = None
    confidence: str | None = None
    status: str | None = None
    paper_id: str | None = None
    paper_title: str | None = None
    section: str | None = None
    gap_status: str | None = None
    categories: list[str] = Field(default_factory=list)


class ClaimGraphEdgeOut(BaseModel):
    source: str
    target: str
    relationship: Literal["supports", "contradicts", "contextualizes", "addresses_gap"]


class ClaimEvidenceGraphOut(BaseModel):
    nodes: list[ClaimGraphNodeOut]
    edges: list[ClaimGraphEdgeOut]


class GapDensityBucketOut(BaseModel):
    category: str
    gap_count: int


class GapDensityOut(BaseModel):
    buckets: list[GapDensityBucketOut]
```

- [ ] **Step 4: Add routes**

In `src/researchbridge/api/assessment_routes.py`, add to the `from researchbridge.api.schemas import (...)` block (alphabetical, matching existing style):

```python
from researchbridge.api.schemas import (
    ClaimEvidenceGraphOut,
    ClaimGraphEdgeOut,
    ClaimGraphNodeOut,
    GapDensityBucketOut,
    GapDensityOut,
    GraphEdgeOut,
    GraphNodeOut,
    ResearchAssessmentCreate,
    ResearchAssessmentHistoryItem,
    ResearchAssessmentOut,
    ResearchAssessmentReview,
    ResearchAssessmentSummaryOut,
    ResearchAssessmentSummaryPage,
    SimilarityGraphOut,
)
```

Add two more imports next to the existing `from researchbridge.assessment.graph import build_similarity_graph` line:

```python
from researchbridge.assessment.claim_graph import build_claim_evidence_graph
from researchbridge.gaps.density import compute_gap_density
```

Add the two routes immediately after the existing `/graph` route (after line 317):

```python
@router.get("/{assessment_id}/claim-graph", response_model=ClaimEvidenceGraphOut)
def get_assessment_claim_graph(
    assessment_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> ClaimEvidenceGraphOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    graph = build_claim_evidence_graph(session, assessment)

    return ClaimEvidenceGraphOut(
        nodes=[
            ClaimGraphNodeOut(
                id=node.id, kind=node.kind, label=node.label, claim_type=node.claim_type,
                confidence=node.confidence, status=node.status, paper_id=node.paper_id,
                paper_title=node.paper_title, section=node.section, gap_status=node.gap_status,
                categories=node.categories,
            )
            for node in graph.nodes
        ],
        edges=[
            ClaimGraphEdgeOut(source=edge.source, target=edge.target, relationship=edge.relationship)
            for edge in graph.edges
        ],
    )


@router.get("/{assessment_id}/gap-density", response_model=GapDensityOut)
def get_gap_density(
    assessment_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> GapDensityOut:
    assessment = session.get(ResearchAssessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id}")

    buckets = compute_gap_density(session)

    return GapDensityOut(
        buckets=[GapDensityBucketOut(category=b.category, gap_count=b.gap_count) for b in buckets]
    )
```

(`assessment_id` is required in the `gap-density` path to keep both routes on the same resource, but the response is corpus-wide, per the spec's resolved scope — the assessment lookup only gates existence, its own gap isn't otherwise used.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_claim_evidence_graph_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full backend suite for this touched area**

Run: `pytest tests/test_claim_graph.py tests/test_gap_density.py tests/test_claim_evidence_graph_api.py tests/test_assessment_graph_api.py -v`
Expected: PASS (all tests, including the pre-existing similarity-graph tests, confirming no regression on the shared route file)

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/assessment_routes.py tests/test_claim_evidence_graph_api.py
git commit -m "$(cat <<'EOF'
feat(api): add claim-graph and gap-density endpoints

GET /api/assessments/{id}/claim-graph and .../gap-density, following
the existing /graph route's no-auth, 404-on-missing-assessment
convention.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Frontend dependencies and API client

**Files:**
- Modify: `frontend/package.json` (via `npm install`)
- Modify: `frontend/lib/assessmentApi.ts` (add types after `GraphData`/before `ReviewFilter`, i.e. after line 158; add two methods to the `assessmentApi` object after the `graph:` entry)

**Interfaces:**
- Produces: `ClaimGraphNode`, `ClaimGraphEdge`, `ClaimEvidenceGraphData`, `GapDensityBucket`, `GapDensityData` types, and `assessmentApi.claimGraph(id)` / `assessmentApi.gapDensity(id)`. Every later frontend task imports these from `@/lib/assessmentApi`.

This task is additive-only (new types + thin fetch wrappers matching the existing `graph:` entry exactly) — no new branching logic to drive with a red/green cycle, so verification is "install, add code, confirm the existing suite still passes" rather than a new failing test.

- [ ] **Step 1: Install dependencies**

```bash
cd frontend && npm install @xyflow/react @dagrejs/dagre
```

- [ ] **Step 2: Add types to `frontend/lib/assessmentApi.ts`**

Insert after line 158 (`export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[]; };`):

```ts
export type ClaimGraphNode = {
  id: string;
  kind: "claim" | "evidence" | "gap";
  label: string;
  claim_type:
    | "fact"
    | "inference"
    | "hypothesis"
    | "opportunity"
    | "speculation"
    | null;
  confidence: string | null;
  status: "pending" | "approved" | "rejected" | null;
  paper_id: string | null;
  paper_title: string | null;
  section: string | null;
  gap_status: "strong_gap" | "potential_gap" | "known_limitation" | null;
  categories: string[];
};

export type ClaimGraphEdge = {
  source: string;
  target: string;
  relationship: "supports" | "contradicts" | "contextualizes" | "addresses_gap";
};

export type ClaimEvidenceGraphData = {
  nodes: ClaimGraphNode[];
  edges: ClaimGraphEdge[];
};

export type GapDensityBucket = {
  category: string;
  gap_count: number;
};

export type GapDensityData = {
  buckets: GapDensityBucket[];
};
```

- [ ] **Step 3: Add methods to `assessmentApi`**

In the `export const assessmentApi = { ... }` object, immediately after the existing `graph: (id: string) => request<GraphData>(...)` entry:

```ts
  claimGraph: (id: string) =>
    request<ClaimEvidenceGraphData>(`/api/assessments/${id}/claim-graph`),
  gapDensity: (id: string) =>
    request<GapDensityData>(`/api/assessments/${id}/gap-density`),
```

- [ ] **Step 4: Run the existing suite to confirm nothing broke**

Run: `cd frontend && npm test`
Expected: PASS (all existing tests — this task added no new behavior)

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/lib/assessmentApi.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add @xyflow/react + @dagrejs/dagre and claim-graph API client

New deps for the interactive claim & evidence graph (stable DAG layout
with custom node cards and a side inspector - a poor fit for the
existing force-directed react-force-graph-2d). Adds thin fetch
wrappers for the two new backend endpoints, matching the existing
graph() method's shape.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Dagre layout helper

**Files:**
- Create: `frontend/lib/graphLayout.ts`
- Test: `frontend/__tests__/graphLayout.test.ts`

**Interfaces:**
- Consumes: `ClaimGraphNode`, `ClaimGraphEdge` types from Task 4 (`@/lib/assessmentApi`).
- Produces: `layoutClaimGraph(nodes: ClaimGraphNode[], edges: ClaimGraphEdge[]): { id: string; x: number; y: number }[]`. Task 11 consumes this.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { layoutClaimGraph } from "@/lib/graphLayout";
import type { ClaimGraphEdge, ClaimGraphNode } from "@/lib/assessmentApi";

function node(id: string, kind: ClaimGraphNode["kind"]): ClaimGraphNode {
  return {
    id,
    kind,
    label: id,
    claim_type: null,
    confidence: null,
    status: null,
    paper_id: null,
    paper_title: null,
    section: null,
    gap_status: null,
    categories: [],
  };
}

describe("layoutClaimGraph", () => {
  it("positions evidence above the claim it supports, and the claim above the gap it addresses", () => {
    const nodes = [node("claim-1", "claim"), node("evidence-1", "evidence"), node("gap-1", "gap")];
    const edges: ClaimGraphEdge[] = [
      { source: "claim-1", target: "evidence-1", relationship: "supports" },
      { source: "claim-1", target: "gap-1", relationship: "addresses_gap" },
    ];

    const positions = layoutClaimGraph(nodes, edges);
    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));

    expect(byId["evidence-1"].y).toBeLessThan(byId["claim-1"].y);
    expect(byId["claim-1"].y).toBeLessThan(byId["gap-1"].y);
  });

  it("gives every node a finite position even with no edges", () => {
    const nodes = [node("claim-1", "claim")];

    const positions = layoutClaimGraph(nodes, []);

    expect(positions).toHaveLength(1);
    expect(Number.isFinite(positions[0].x)).toBe(true);
    expect(Number.isFinite(positions[0].y)).toBe(true);
  });

  it("returns one position per input node", () => {
    const nodes = [node("a", "claim"), node("b", "evidence"), node("c", "gap")];

    const positions = layoutClaimGraph(nodes, []);

    expect(positions.map((p) => p.id).sort()).toEqual(["a", "b", "c"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/graphLayout.test.ts`
Expected: FAIL with a module-not-found error for `@/lib/graphLayout`

- [ ] **Step 3: Write the implementation**

```ts
import dagre from "@dagrejs/dagre";
import type { ClaimGraphEdge, ClaimGraphNode } from "./assessmentApi";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 90;

export type LayoutedPosition = { id: string; x: number; y: number };

/**
 * Layered top-to-bottom DAG: evidence ranks above the claims it backs,
 * claims rank above the gap they address. Edges are stored claim -> evidence
 * and claim -> gap (see ClaimGraphEdge), so the "supports"/"contradicts"/
 * "contextualizes" edges are fed to dagre reversed (evidence -> claim) to
 * get the desired rank order; "addresses_gap" edges are fed as-is
 * (claim -> gap already ranks claims above the gap).
 */
export function layoutClaimGraph(nodes: ClaimGraphNode[], edges: ClaimGraphEdge[]): LayoutedPosition[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 40, ranksep: 90 });

  for (const node of nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    if (edge.relationship === "addresses_gap") {
      g.setEdge(edge.source, edge.target);
    } else {
      g.setEdge(edge.target, edge.source);
    }
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const position = g.node(node.id);
    return { id: node.id, x: position.x, y: position.y };
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/graphLayout.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/graphLayout.ts frontend/__tests__/graphLayout.test.ts
git commit -m "$(cat <<'EOF'
feat(frontend): add dagre layout helper for the claim/evidence graph

Layers evidence above claims and claims above the gap they address by
feeding dagre reversed edges for the supports/contradicts/
contextualizes direction while keeping addresses_gap edges as-is.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Graph node components (Claim, Evidence, Gap)

**Files:**
- Create: `frontend/components/graph/ClaimNode.tsx`
- Create: `frontend/components/graph/EvidenceNode.tsx`
- Create: `frontend/components/graph/GapNode.tsx`
- Test: `frontend/__tests__/ClaimNode.test.tsx`
- Test: `frontend/__tests__/EvidenceNode.test.tsx`
- Test: `frontend/__tests__/GapNode.test.tsx`

**Interfaces:**
- Consumes: `GAP_STATUS_LABELS` from `@/lib/gapsApi` (existing).
- Produces: `ClaimNode` + `ClaimNodeData`/`ClaimFlowNode`; `EvidenceNode` + `EvidenceNodeData`/`EvidenceFlowNode`; `GapNode` + `GapNodeData`/`GapFlowNode`. Task 11 imports all six.

- [ ] **Step 1: Write the failing tests**

`frontend/__tests__/ClaimNode.test.tsx`:
```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { ClaimNode, type ClaimFlowNode } from "@/components/graph/ClaimNode";

function props(data: ClaimFlowNode["data"]): NodeProps<ClaimFlowNode> {
  return { id: "claim-1", type: "claim", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<ClaimFlowNode>;
}

describe("ClaimNode", () => {
  it("renders the claim text, type, confidence, and status", () => {
    render(
      <ReactFlowProvider>
        <ClaimNode {...props({ label: "The method generalizes.", claim_type: "inference", confidence: "medium", status: "pending" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("The method generalizes.")).toBeInTheDocument();
    expect(screen.getByText("claim · inference")).toBeInTheDocument();
    expect(screen.getByText(/medium confidence/)).toBeInTheDocument();
    expect(screen.getByText(/pending/)).toBeInTheDocument();
  });

  it("omits the type suffix when claim_type is null", () => {
    render(
      <ReactFlowProvider>
        <ClaimNode {...props({ label: "A claim.", claim_type: null, confidence: null, status: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("claim")).toBeInTheDocument();
  });
});
```

`frontend/__tests__/EvidenceNode.test.tsx`:
```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { EvidenceNode, type EvidenceFlowNode } from "@/components/graph/EvidenceNode";

function props(data: EvidenceFlowNode["data"]): NodeProps<EvidenceFlowNode> {
  return { id: "evidence-1", type: "evidence", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<EvidenceFlowNode>;
}

describe("EvidenceNode", () => {
  it("renders the quoted passage, paper title, and section", () => {
    render(
      <ReactFlowProvider>
        <EvidenceNode {...props({ label: "a quoted passage", paper_title: "Some Paper", section: "results" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText('"a quoted passage"')).toBeInTheDocument();
    expect(screen.getByText("Some Paper")).toBeInTheDocument();
    expect(screen.getByText("evidence · results")).toBeInTheDocument();
  });

  it("omits the paper title line when paper_title is null", () => {
    render(
      <ReactFlowProvider>
        <EvidenceNode {...props({ label: "a quoted passage", paper_title: null, section: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("evidence")).toBeInTheDocument();
  });
});
```

`frontend/__tests__/GapNode.test.tsx`:
```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { GapNode, type GapFlowNode } from "@/components/graph/GapNode";

function props(data: GapFlowNode["data"]): NodeProps<GapFlowNode> {
  return { id: "gap-1", type: "gap", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<GapFlowNode>;
}

describe("GapNode", () => {
  it("renders the observation and a human-readable gap status label", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "a recurring unaddressed limitation", gap_status: "strong_gap" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("a recurring unaddressed limitation")).toBeInTheDocument();
    expect(screen.getByText("strong gap")).toBeInTheDocument();
  });

  it("renders without a status badge when gap_status is null", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "an observation", gap_status: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("an observation")).toBeInTheDocument();
    expect(screen.queryByText("strong gap")).not.toBeInTheDocument();
  });

  it("applies stronger emphasis when highlighted by the density overlay", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "an observation", gap_status: null, highlighted: true })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("an observation").closest("div")?.className).toContain("border-[var(--ink)]");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run __tests__/ClaimNode.test.tsx __tests__/EvidenceNode.test.tsx __tests__/GapNode.test.tsx`
Expected: FAIL — modules `@/components/graph/ClaimNode`, `EvidenceNode`, `GapNode` don't exist yet

- [ ] **Step 3: Write the implementations**

`frontend/components/graph/ClaimNode.tsx`:
```tsx
"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";

export type ClaimNodeData = {
  label: string;
  claim_type: string | null;
  confidence: string | null;
  status: string | null;
};

export type ClaimFlowNode = Node<ClaimNodeData, "claim">;

export function ClaimNode({ data }: NodeProps<ClaimFlowNode>) {
  return (
    <div className="min-w-[200px] max-w-[240px] rounded-[3px] border border-[var(--ink)] px-3 py-2">
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        claim{data.claim_type ? ` · ${data.claim_type}` : ""}
      </span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] leading-snug text-[var(--ink)]">{data.label}</p>
      <div className="mt-1 text-[0.6875rem] text-[var(--ink-faint)]">
        {data.confidence ? `${data.confidence} confidence` : null}
        {data.status ? ` · ${data.status}` : null}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
```

`frontend/components/graph/EvidenceNode.tsx`:
```tsx
"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";

export type EvidenceNodeData = {
  label: string;
  paper_title: string | null;
  section: string | null;
};

export type EvidenceFlowNode = Node<EvidenceNodeData, "evidence">;

export function EvidenceNode({ data }: NodeProps<EvidenceFlowNode>) {
  return (
    <div className="min-w-[200px] max-w-[240px] rounded-[3px] border border-[var(--rule)] px-3 py-2">
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        evidence{data.section ? ` · ${data.section}` : ""}
      </span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] italic leading-snug text-[var(--ink-soft)]">
        &quot;{data.label}&quot;
      </p>
      {data.paper_title && (
        <div className="mt-1 truncate text-[0.6875rem] text-[var(--ink-faint)]">{data.paper_title}</div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
```

`frontend/components/graph/GapNode.tsx`:
```tsx
"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";
import { GAP_STATUS_LABELS } from "@/lib/gapsApi";

export type GapNodeData = {
  label: string;
  gap_status: "strong_gap" | "potential_gap" | "known_limitation" | null;
  highlighted?: boolean;
};

export type GapFlowNode = Node<GapNodeData, "gap">;

export function GapNode({ data }: NodeProps<GapFlowNode>) {
  return (
    <div
      className={`min-w-[200px] max-w-[240px] rounded-[3px] border-2 px-3 py-2 ${
        data.highlighted ? "border-[var(--ink)]" : "border-[var(--ink-soft)]"
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">candidate gap</span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] leading-snug text-[var(--ink)]">{data.label}</p>
      {data.gap_status && (
        <span className="eyebrow mt-1 inline-block rounded-[2px] border border-[var(--rule)] px-2 py-0.5 text-[0.625rem] text-[var(--ink-soft)]">
          {GAP_STATUS_LABELS[data.gap_status]}
        </span>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run __tests__/ClaimNode.test.tsx __tests__/EvidenceNode.test.tsx __tests__/GapNode.test.tsx`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/graph/ClaimNode.tsx frontend/components/graph/EvidenceNode.tsx frontend/components/graph/GapNode.tsx frontend/__tests__/ClaimNode.test.tsx frontend/__tests__/EvidenceNode.test.tsx frontend/__tests__/GapNode.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): add Claim/Evidence/Gap node components

Distinguishes node kind by border weight/style, not color - no
traffic-light palette, consistent with the app's existing decision not
to color-code confidence. Reuses GAP_STATUS_LABELS from gapsApi.ts for
the gap status badge.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Relationship edge component

**Files:**
- Create: `frontend/components/graph/RelationshipEdge.tsx`
- Test: `frontend/__tests__/RelationshipEdge.test.tsx`

**Interfaces:**
- Produces: `RelationshipEdge` component (registered under edge type `"relationship"`, reads `data.relationship`). Task 11 imports this.

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import { RelationshipEdge } from "@/components/graph/RelationshipEdge";

function props(relationship: string): EdgeProps {
  return {
    id: "e1", source: "a", target: "b", sourceX: 0, sourceY: 0, targetX: 100, targetY: 100,
    sourcePosition: "bottom", targetPosition: "top", data: { relationship },
  } as unknown as EdgeProps;
}

function renderEdge(relationship: string) {
  return render(
    <ReactFlowProvider>
      <svg>
        <RelationshipEdge {...props(relationship)} />
      </svg>
    </ReactFlowProvider>,
  );
}

describe("RelationshipEdge", () => {
  it("renders a solid line for a supports relationship", () => {
    const { container } = renderEdge("supports");

    expect(container.querySelector("path")?.getAttribute("stroke-dasharray")).toBeNull();
  });

  it("renders a dashed line for a contradicts relationship", () => {
    const { container } = renderEdge("contradicts");

    expect(container.querySelector("path")?.getAttribute("stroke-dasharray")).toBe("6 4");
  });

  it("renders a dotted line for a contextualizes relationship", () => {
    const { container } = renderEdge("contextualizes");

    expect(container.querySelector("path")?.getAttribute("stroke-dasharray")).toBe("2 4");
  });

  it("renders a distinct dash pattern for an addresses_gap relationship", () => {
    const { container } = renderEdge("addresses_gap");

    expect(container.querySelector("path")?.getAttribute("stroke-dasharray")).toBe("10 4");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/RelationshipEdge.test.tsx`
Expected: FAIL — module `@/components/graph/RelationshipEdge` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
"use client";

import { BaseEdge, getBezierPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

const DASH_BY_RELATIONSHIP: Record<string, string | undefined> = {
  supports: undefined,
  contradicts: "6 4",
  contextualizes: "2 4",
  addresses_gap: "10 4",
};

export function RelationshipEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const relationship = (data?.relationship as string | undefined) ?? "supports";
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{ stroke: "var(--ink-soft)", strokeDasharray: DASH_BY_RELATIONSHIP[relationship] }}
    />
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/RelationshipEdge.test.tsx`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/graph/RelationshipEdge.tsx frontend/__tests__/RelationshipEdge.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): add relationship-styled edge for the claim graph

Solid/dashed/dotted line styles distinguish supports/contradicts/
contextualizes/addresses_gap - all in ink-soft, no red/green.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Side inspector drawer

**Files:**
- Create: `frontend/components/graph/InspectorDrawer.tsx`
- Test: `frontend/__tests__/InspectorDrawer.test.tsx`

**Interfaces:**
- Consumes: `ClaimGraphNode` type from Task 4, `GAP_STATUS_LABELS` from `@/lib/gapsApi`.
- Produces: `InspectorDrawer` component (`props: { node: ClaimGraphNode | null; onClose: () => void }`). Task 11 imports this.

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InspectorDrawer } from "@/components/graph/InspectorDrawer";
import type { ClaimGraphNode } from "@/lib/assessmentApi";

function claimNode(overrides: Partial<ClaimGraphNode> = {}): ClaimGraphNode {
  return {
    id: "c1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
    confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
    gap_status: null, categories: [], ...overrides,
  };
}

describe("InspectorDrawer", () => {
  it("renders nothing when no node is selected", () => {
    const { container } = render(<InspectorDrawer node={null} onClose={() => {}} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("shows claim details for a claim node", () => {
    render(<InspectorDrawer node={claimNode()} onClose={() => {}} />);

    expect(screen.getByText("The method generalizes.")).toBeInTheDocument();
    expect(screen.getByText("confidence: medium")).toBeInTheDocument();
    expect(screen.getByText("status: pending")).toBeInTheDocument();
  });

  it("shows evidence details for an evidence node", () => {
    render(
      <InspectorDrawer
        node={claimNode({ kind: "evidence", label: "a quoted passage", paper_title: "Some Paper", section: "results" })}
        onClose={() => {}}
      />,
    );

    expect(screen.getByText("paper: Some Paper")).toBeInTheDocument();
    expect(screen.getByText("section: results")).toBeInTheDocument();
  });

  it("shows a gap status badge for a gap node", () => {
    render(
      <InspectorDrawer node={claimNode({ kind: "gap", label: "an observation", gap_status: "strong_gap" })} onClose={() => {}} />,
    );

    expect(screen.getByText("strong gap")).toBeInTheDocument();
  });

  it("calls onClose when the close control is clicked", () => {
    const onClose = vi.fn();
    render(<InspectorDrawer node={claimNode()} onClose={onClose} />);

    fireEvent.click(screen.getByRole("button", { name: "close" }));

    expect(onClose).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/InspectorDrawer.test.tsx`
Expected: FAIL — module `@/components/graph/InspectorDrawer` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
"use client";

import type { ClaimGraphNode } from "@/lib/assessmentApi";
import { GAP_STATUS_LABELS } from "@/lib/gapsApi";

export type InspectorDrawerProps = {
  node: ClaimGraphNode | null;
  onClose: () => void;
};

export function InspectorDrawer({ node, onClose }: InspectorDrawerProps) {
  if (!node) {
    return null;
  }

  return (
    <aside className="fixed right-0 top-0 h-full w-[340px] overflow-y-auto border-l border-[var(--rule)] p-6">
      <button type="button" onClick={onClose} className="eyebrow text-[var(--ink-faint)] hover:text-[var(--ink)]">
        close
      </button>
      <span className="eyebrow mt-4 block text-[0.625rem] text-[var(--ink-faint)]">{node.kind}</span>
      <p className="mt-2 text-[0.9375rem] leading-relaxed text-[var(--ink)]">{node.label}</p>

      {node.kind === "claim" && (
        <dl className="mt-4 space-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
          {node.claim_type && <div>type: {node.claim_type}</div>}
          {node.confidence && <div>confidence: {node.confidence}</div>}
          {node.status && <div>status: {node.status}</div>}
        </dl>
      )}

      {node.kind === "evidence" && (
        <dl className="mt-4 space-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
          {node.paper_title && <div>paper: {node.paper_title}</div>}
          {node.section && <div>section: {node.section}</div>}
        </dl>
      )}

      {node.kind === "gap" && node.gap_status && (
        <span className="eyebrow mt-4 inline-block rounded-[2px] border border-[var(--rule)] px-2 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]">
          {GAP_STATUS_LABELS[node.gap_status]}
        </span>
      )}
    </aside>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/InspectorDrawer.test.tsx`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/graph/InspectorDrawer.tsx frontend/__tests__/InspectorDrawer.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): add side inspector drawer for the claim graph

Shows full text and kind-specific detail (claim type/confidence/
status, evidence paper/section, gap status) for whichever node is
selected.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Filter toolbar

**Files:**
- Create: `frontend/components/graph/FilterToolbar.tsx`
- Test: `frontend/__tests__/FilterToolbar.test.tsx`

**Interfaces:**
- Produces: `KindFilter` (`{ claim: boolean; evidence: boolean; gap: boolean }`), `ConfidenceFilter` (`{ high: boolean; medium: boolean; low: boolean }`), `FilterToolbar` component (`props: { kinds: KindFilter; onKindsChange: (k: KindFilter) => void; confidence: ConfidenceFilter; onConfidenceChange: (c: ConfidenceFilter) => void }`). Task 11 imports all three.

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { FilterToolbar } from "@/components/graph/FilterToolbar";

const allKinds = { claim: true, evidence: true, gap: true };
const allConfidence = { high: true, medium: true, low: true };

describe("FilterToolbar", () => {
  it("renders a checked checkbox for each node kind and confidence level", () => {
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={() => {}} confidence={allConfidence} onConfidenceChange={() => {}} />,
    );

    expect(screen.getByRole("checkbox", { name: "claims" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "evidence" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "gaps" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "high" })).toBeChecked();
  });

  it("calls onKindsChange with only the toggled kind flipped", () => {
    const onKindsChange = vi.fn();
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={onKindsChange} confidence={allConfidence} onConfidenceChange={() => {}} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "gaps" }));

    expect(onKindsChange).toHaveBeenCalledWith({ ...allKinds, gap: false });
  });

  it("calls onConfidenceChange with only the toggled level flipped", () => {
    const onConfidenceChange = vi.fn();
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={() => {}} confidence={allConfidence} onConfidenceChange={onConfidenceChange} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "low" }));

    expect(onConfidenceChange).toHaveBeenCalledWith({ ...allConfidence, low: false });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/FilterToolbar.test.tsx`
Expected: FAIL — module `@/components/graph/FilterToolbar` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
"use client";

export type KindFilter = { claim: boolean; evidence: boolean; gap: boolean };
export type ConfidenceFilter = { high: boolean; medium: boolean; low: boolean };

export type FilterToolbarProps = {
  kinds: KindFilter;
  onKindsChange: (kinds: KindFilter) => void;
  confidence: ConfidenceFilter;
  onConfidenceChange: (confidence: ConfidenceFilter) => void;
};

const KIND_LABELS: Record<keyof KindFilter, string> = { claim: "claims", evidence: "evidence", gap: "gaps" };
const CONFIDENCE_LABELS: Record<keyof ConfidenceFilter, string> = { high: "high", medium: "medium", low: "low" };

export function FilterToolbar({ kinds, onKindsChange, confidence, onConfidenceChange }: FilterToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-[var(--rule-soft)] pb-3">
      <div className="flex items-center gap-2">
        {(Object.keys(KIND_LABELS) as (keyof KindFilter)[]).map((kind) => (
          <label key={kind} className="eyebrow flex items-center gap-1 text-[0.6875rem] text-[var(--ink-soft)]">
            <input
              type="checkbox"
              checked={kinds[kind]}
              onChange={(e) => onKindsChange({ ...kinds, [kind]: e.target.checked })}
            />
            {KIND_LABELS[kind]}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2">
        {(Object.keys(CONFIDENCE_LABELS) as (keyof ConfidenceFilter)[]).map((level) => (
          <label key={level} className="eyebrow flex items-center gap-1 text-[0.6875rem] text-[var(--ink-soft)]">
            <input
              type="checkbox"
              checked={confidence[level]}
              onChange={(e) => onConfidenceChange({ ...confidence, [level]: e.target.checked })}
            />
            {CONFIDENCE_LABELS[level]}
          </label>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/FilterToolbar.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/graph/FilterToolbar.tsx frontend/__tests__/FilterToolbar.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): add node-kind and confidence filter toolbar

Client-side checkbox filters for the claim graph - no refetch per
toggle, filtering happens over the already-fetched graph.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Gap density overlay

**Files:**
- Create: `frontend/components/graph/GapDensityOverlay.tsx`
- Test: `frontend/__tests__/GapDensityOverlay.test.tsx`

**Interfaces:**
- Consumes: `assessmentApi.gapDensity`, `GapDensityBucket` type from Task 4.
- Produces: `GapDensityOverlay` component (`props: { assessmentId: string; gapCategories: string[]; onHighlightChange: (highlighted: boolean) => void }`). Task 11 imports this.

- [ ] **Step 1: Write the failing test**

```tsx
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GapDensityOverlay } from "@/components/graph/GapDensityOverlay";
import { assessmentApi } from "@/lib/assessmentApi";

describe("GapDensityOverlay", () => {
  afterEach(() => vi.restoreAllMocks());

  it("does not report a highlight before the toggle is enabled", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [{ category: "Machine Learning", gap_count: 10 }],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Machine Learning"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    expect(onHighlightChange).toHaveBeenLastCalledWith(false);
  });

  it("reports a highlight once enabled, when the gap's category is at or above the average bucket count", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [
        { category: "Machine Learning", gap_count: 10 },
        { category: "Robotics", gap_count: 2 },
      ],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Machine Learning"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() => expect(onHighlightChange).toHaveBeenLastCalledWith(true));
  });

  it("does not report a highlight for a below-average category", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [
        { category: "Machine Learning", gap_count: 10 },
        { category: "Robotics", gap_count: 2 },
      ],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Robotics"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() => expect(onHighlightChange).toHaveBeenLastCalledWith(false));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/GapDensityOverlay.test.tsx`
Expected: FAIL — module `@/components/graph/GapDensityOverlay` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
"use client";

import { useEffect, useState } from "react";
import { assessmentApi, type GapDensityBucket } from "@/lib/assessmentApi";

export type GapDensityOverlayProps = {
  assessmentId: string;
  gapCategories: string[];
  onHighlightChange: (highlighted: boolean) => void;
};

export function GapDensityOverlay({ assessmentId, gapCategories, onHighlightChange }: GapDensityOverlayProps) {
  const [buckets, setBuckets] = useState<GapDensityBucket[] | null>(null);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    assessmentApi.gapDensity(assessmentId).then((data) => setBuckets(data.buckets));
  }, [assessmentId]);

  useEffect(() => {
    if (!enabled || !buckets) {
      onHighlightChange(false);
      return;
    }
    const average = buckets.reduce((sum, b) => sum + b.gap_count, 0) / (buckets.length || 1);
    const highlighted = buckets.some((b) => gapCategories.includes(b.category) && b.gap_count >= average);
    onHighlightChange(highlighted);
  }, [enabled, buckets, gapCategories, onHighlightChange]);

  return (
    <button
      type="button"
      onClick={() => setEnabled((v) => !v)}
      disabled={!buckets}
      className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
        enabled ? "border-[var(--ink)] text-[var(--ink)]" : "border-[var(--rule)] text-[var(--ink-faint)]"
      }`}
    >
      gap density overlay
    </button>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/GapDensityOverlay.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/components/graph/GapDensityOverlay.tsx frontend/__tests__/GapDensityOverlay.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): add gap density overlay toggle

Fetches corpus-wide gap-density buckets once and, when enabled, flags
the graph's gap node as highlighted if its seed paper's category sits
at or above the average bucket count - an ink-intensity highlight, not
a heat-color scale.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Main graph component (assembly)

**Files:**
- Create: `frontend/components/graph/ClaimEvidenceGraph.tsx`
- Test: `frontend/__tests__/ClaimEvidenceGraph.test.tsx`

**Interfaces:**
- Consumes: everything from Tasks 4–10 (`assessmentApi.claimGraph`, `layoutClaimGraph`, `ClaimNode`/`EvidenceNode`/`GapNode`, `RelationshipEdge`, `InspectorDrawer`, `FilterToolbar`, `GapDensityOverlay`).
- Produces: `ClaimEvidenceGraph` component (`props: { assessmentId: string }`). Task 12 imports this.

- [ ] **Step 1: Write the failing test**

```tsx
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ClaimEvidenceGraph } from "@/components/graph/ClaimEvidenceGraph";
import { assessmentApi } from "@/lib/assessmentApi";

describe("ClaimEvidenceGraph", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the claim graph and renders each node's label", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "claim-1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
          confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
          gap_status: null, categories: [],
        },
        {
          id: "evidence-1", kind: "evidence", label: "a quoted passage", claim_type: null, confidence: null,
          status: null, paper_id: "p1", paper_title: "Some Paper", section: "results", gap_status: null,
          categories: [],
        },
      ],
      edges: [{ source: "claim-1", target: "evidence-1", relationship: "supports" }],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getByText("The method generalizes.")).toBeInTheDocument());
    expect(screen.getByText('"a quoted passage"')).toBeInTheDocument();
  });

  it("hides gap nodes when the gaps filter is unchecked", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "gap-1", kind: "gap", label: "a recurring unaddressed limitation", claim_type: null,
          confidence: null, status: null, paper_id: null, paper_title: null, section: null,
          gap_status: "strong_gap", categories: [],
        },
      ],
      edges: [],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getByText("a recurring unaddressed limitation")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("checkbox", { name: "gaps" }));

    expect(screen.queryByText("a recurring unaddressed limitation")).not.toBeInTheDocument();
  });

  it("opens the inspector drawer with the clicked node's detail", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "claim-1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
          confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
          gap_status: null, categories: [],
        },
      ],
      edges: [],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getAllByText("The method generalizes.")).toHaveLength(1));

    fireEvent.click(screen.getByText("The method generalizes."));

    await waitFor(() => expect(screen.getAllByText("The method generalizes.")).toHaveLength(2));
    expect(screen.getByText("confidence: medium")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run __tests__/ClaimEvidenceGraph.test.tsx`
Expected: FAIL — module `@/components/graph/ClaimEvidenceGraph` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { assessmentApi, type ClaimGraphEdge, type ClaimGraphNode } from "@/lib/assessmentApi";
import { layoutClaimGraph } from "@/lib/graphLayout";
import { ClaimNode, type ClaimFlowNode } from "@/components/graph/ClaimNode";
import { EvidenceNode, type EvidenceFlowNode } from "@/components/graph/EvidenceNode";
import { GapNode, type GapFlowNode } from "@/components/graph/GapNode";
import { RelationshipEdge } from "@/components/graph/RelationshipEdge";
import { InspectorDrawer } from "@/components/graph/InspectorDrawer";
import { FilterToolbar, type ConfidenceFilter, type KindFilter } from "@/components/graph/FilterToolbar";
import { GapDensityOverlay } from "@/components/graph/GapDensityOverlay";

const nodeTypes = { claim: ClaimNode, evidence: EvidenceNode, gap: GapNode };
const edgeTypes = { relationship: RelationshipEdge };

export type ClaimEvidenceGraphProps = { assessmentId: string };

function toFlowNode(node: ClaimGraphNode, position: { x: number; y: number }, gapHighlighted: boolean): Node {
  if (node.kind === "claim") {
    const flowNode: ClaimFlowNode = {
      id: node.id,
      type: "claim",
      position,
      data: { label: node.label, claim_type: node.claim_type, confidence: node.confidence, status: node.status },
    };
    return flowNode;
  }
  if (node.kind === "evidence") {
    const flowNode: EvidenceFlowNode = {
      id: node.id,
      type: "evidence",
      position,
      data: { label: node.label, paper_title: node.paper_title, section: node.section },
    };
    return flowNode;
  }
  const flowNode: GapFlowNode = {
    id: node.id,
    type: "gap",
    position,
    data: { label: node.label, gap_status: node.gap_status, highlighted: gapHighlighted },
  };
  return flowNode;
}

function toFlowEdge(edge: ClaimGraphEdge, index: number): Edge {
  return {
    id: `${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    type: "relationship",
    data: { relationship: edge.relationship },
  };
}

export function ClaimEvidenceGraph({ assessmentId }: ClaimEvidenceGraphProps) {
  const [nodes, setNodes] = useState<ClaimGraphNode[]>([]);
  const [edges, setEdges] = useState<ClaimGraphEdge[]>([]);
  const [selected, setSelected] = useState<ClaimGraphNode | null>(null);
  const [kindFilter, setKindFilter] = useState<KindFilter>({ claim: true, evidence: true, gap: true });
  const [confidenceFilter, setConfidenceFilter] = useState<ConfidenceFilter>({ high: true, medium: true, low: true });
  const [gapHighlighted, setGapHighlighted] = useState(false);

  useEffect(() => {
    assessmentApi.claimGraph(assessmentId).then((data) => {
      setNodes(data.nodes);
      setEdges(data.edges);
    });
  }, [assessmentId]);

  const visibleNodes = useMemo(
    () =>
      nodes.filter((node) => {
        if (!kindFilter[node.kind]) return false;
        if (node.kind === "claim" && node.confidence && node.confidence in confidenceFilter) {
          return confidenceFilter[node.confidence as keyof ConfidenceFilter];
        }
        return true;
      }),
    [nodes, kindFilter, confidenceFilter],
  );

  const visibleIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () => edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)),
    [edges, visibleIds],
  );

  const positions = useMemo(() => layoutClaimGraph(visibleNodes, visibleEdges), [visibleNodes, visibleEdges]);
  const positionById = useMemo(() => Object.fromEntries(positions.map((p) => [p.id, p])), [positions]);

  const gapCategories = useMemo(() => nodes.find((n) => n.kind === "gap")?.categories ?? [], [nodes]);

  const flowNodes = visibleNodes.map((node) =>
    toFlowNode(node, positionById[node.id] ?? { x: 0, y: 0 }, node.kind === "gap" && gapHighlighted),
  );
  const flowEdges = visibleEdges.map(toFlowEdge);

  return (
    <div className="mt-8">
      <FilterToolbar kinds={kindFilter} onKindsChange={setKindFilter} confidence={confidenceFilter} onConfidenceChange={setConfidenceFilter} />
      <div className="mt-3">
        <GapDensityOverlay assessmentId={assessmentId} gapCategories={gapCategories} onHighlightChange={setGapHighlighted} />
      </div>
      <div style={{ height: 480 }} className="mt-4 border border-[var(--rule-soft)]">
        <ReactFlowProvider>
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodeClick={(_, node) => setSelected(nodes.find((n) => n.id === node.id) ?? null)}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
      <InspectorDrawer node={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run __tests__/ClaimEvidenceGraph.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full frontend suite**

Run: `cd frontend && npm test`
Expected: PASS (all tests, including the new graph component tests and the pre-existing suite)

- [ ] **Step 6: Commit**

```bash
git add frontend/components/graph/ClaimEvidenceGraph.tsx frontend/__tests__/ClaimEvidenceGraph.test.tsx
git commit -m "$(cat <<'EOF'
feat(frontend): assemble the interactive claim & evidence graph

Fetches the claim graph, applies kind/confidence filters, lays it out
with dagre, and wires up the inspector drawer and gap density overlay.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Integrate into the assessment page

**Files:**
- Modify: `frontend/app/assessments/[id]/page.tsx`

**Interfaces:**
- Consumes: `ClaimEvidenceGraph` from Task 11.

No new test file — this task wires an already-tested component into an already-tested page shell. Verification is a manual dev-server check plus the existing suite staying green.

- [ ] **Step 1: Add the import**

In `frontend/app/assessments/[id]/page.tsx`, add after line 7 (`import { SimilarityGraph } from "@/components/SimilarityGraph";`):

```tsx
import { ClaimEvidenceGraph } from "@/components/graph/ClaimEvidenceGraph";
```

- [ ] **Step 2: Render it below the existing similarity graph**

Replace lines 41–46:

```tsx
      {assessment && (
        <div className="pt-12">
          <AssessmentReport assessment={assessment} onAssessmentUpdated={setAssessment} />
          <SimilarityGraph assessmentId={id} />
        </div>
      )}
```

with:

```tsx
      {assessment && (
        <div className="pt-12">
          <AssessmentReport assessment={assessment} onAssessmentUpdated={setAssessment} />
          <SimilarityGraph assessmentId={id} />
          <ClaimEvidenceGraph assessmentId={id} />
        </div>
      )}
```

- [ ] **Step 3: Run the full frontend suite**

Run: `cd frontend && npm test`
Expected: PASS (no regressions)

- [ ] **Step 4: Manual verification in the dev server**

Run: `cd frontend && npm run dev` (with the backend running separately), open an existing assessment's page, and confirm the claim & evidence graph section renders below the similarity graph — nodes appear, clicking a node opens the inspector drawer, the filter toolbar toggles node visibility, and the gap density overlay button toggles the gap node's border emphasis.

- [ ] **Step 5: Commit**

```bash
git add "frontend/app/assessments/[id]/page.tsx"
git commit -m "$(cat <<'EOF'
feat(frontend): show the claim & evidence graph on the assessment page

Adds it as a new section below the existing similarity graph, matching
the page's existing plain vertical-stack layout (no tab component
exists in this codebase).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** claim/evidence/gap nodes (Task 1, 6), SUPPORTS/CONTRADICTS/CONTEXTUALIZES/ADDRESSES_GAP edges (Task 1, 7), `/claim-graph` and `/gap-density` routes (Task 3), dagre layout (Task 5), inspector drawer (Task 8), filter toolbar (Task 9), gap density overlay (Task 10), assessment-page integration (Task 12) — every spec section maps to a task.
- **Placeholder scan:** no TBD/TODO; every step has literal code.
- **Type consistency checked:** `ClaimGraphNode`/`ClaimGraphEdge` field names and the `categories` field (added during planning to let `GapDensityOverlay` know the gap's own domain categories — the spec's data model didn't originally surface this on the node, but omitting it would leave the density overlay with no way to evaluate the one gap in scope) are consistent across Task 1 (Python dataclass), Task 3 (Pydantic schema + route mapping), and Task 4 (TypeScript type) through to Task 11's usage.
