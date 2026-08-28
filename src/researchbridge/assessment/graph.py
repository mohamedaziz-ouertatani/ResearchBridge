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
    #
    # Deliberately no `Paper.excluded_at.is_(None)` filter here, unlike the
    # retrieval-shaped queries in embedding/search.py, gaps/batch.py, and
    # api/routes.py. Those filter live corpus state for new retrieval; this
    # graph instead reconstructs what THIS assessment was actually compared
    # against at build time, from its already-stored retrieved_paper_ids. A
    # paper curated out afterward doesn't retroactively change what was
    # compared, so it still belongs in the graph.
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
    for row in rows:
        distance = 1.0 - _dot(query_vector, row.vector)
        paper_nodes.append(
            GraphNode(
                id=str(row.id), type="paper", title=row.title, distance_to_input=distance,
                claim_counts=claim_counts_by_paper.get(row.id, {}),
            )
        )

    # Star topology only: input<->paper edges, deliberately no paper<->paper
    # pairwise edges - a full graph of C(n,2) cross-edges reads as noise once
    # there are more than a handful of retrieved papers, and every distance
    # a reader actually wants ("how close is this paper to what I proposed")
    # is already on the input edge.
    edges = [GraphEdge(source="input", target=node.id, distance=node.distance_to_input) for node in paper_nodes]

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
