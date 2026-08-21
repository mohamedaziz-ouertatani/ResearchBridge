"""Orchestrates Sec 32's implicit-gap pipeline for one seed paper:

    Related Papers -> Paper Comparison (limitations/research_gap claims)
    -> Cross-Paper Synthesis -> Candidate Research Gaps

"Related Papers" reuses the Week 6 paper-to-paper similarity search
(embedding/search.py) rather than the retrieval package's text-query
Retriever protocol - this is paper-to-paper, not query-to-paper, and
that's already built, evaluated, and correct for the job.

Returns drafts only. Nothing is persisted or shown as validated here -
gaps/persistence.py writes drafts as status="pending" CandidateGap rows,
and only a human review changes that (Sec 35/44).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import ExtractedClaim, Paper
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import find_similar_to_paper
from researchbridge.gaps.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    RELEVANT_CLAIM_TYPES,
    ClaimRecord,
    find_recurring_patterns,
)

DETECTION_METHOD = "cluster-v1"


@dataclass
class CandidateGapDraft:
    seed_paper_id: uuid.UUID
    observation: str
    contributing_paper_count: int
    evidence_ids: list[uuid.UUID]


def detect_candidate_gaps(
    session: Session,
    seed_paper_id: uuid.UUID,
    embedder: Embedder,
    top_k: int = 15,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[CandidateGapDraft]:
    if session.get(Paper, seed_paper_id) is None:
        raise ValueError(f"no paper with id {seed_paper_id}")

    # find_similar_to_paper excludes the seed itself; add it back in, since
    # its own stated limitations/gaps belong in the same neighborhood pool
    related = find_similar_to_paper(session, seed_paper_id, embedder.model_name, top_k)
    neighborhood_ids = [seed_paper_id] + [paper.id for paper, _distance in related]

    claims = _load_claims(session, neighborhood_ids)
    clusters = find_recurring_patterns(claims, embedder, min_cluster_size, similarity_threshold)

    return [
        CandidateGapDraft(
            seed_paper_id=seed_paper_id,
            observation=_render_observation(cluster),
            contributing_paper_count=cluster.contributing_paper_count,
            evidence_ids=[m.evidence_id for m in cluster.members],
        )
        for cluster in clusters
    ]


def _load_claims(session: Session, paper_ids: list[uuid.UUID]) -> list[ClaimRecord]:
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.evidence_id, ExtractedClaim.text).where(
            ExtractedClaim.paper_id.in_(paper_ids), ExtractedClaim.claim_type.in_(RELEVANT_CLAIM_TYPES)
        )
    ).all()
    return [ClaimRecord(paper_id=paper_id, evidence_id=evidence_id, text=text) for paper_id, evidence_id, text in rows]


def _render_observation(cluster) -> str:
    return (
        f"Recurring pattern across {cluster.contributing_paper_count} related papers "
        f"(inference, not stated by any single author): \"{cluster.representative_text}\""
    )
