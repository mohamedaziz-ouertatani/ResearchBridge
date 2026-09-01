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
from researchbridge.gaps.signals import cosine_similarity

DETECTION_METHOD = "cluster-v1"

CONTRIBUTION_CLAIM_TYPES = ("main_contribution", "results")


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


@dataclass
class _GapClaimRow:
    paper_id: uuid.UUID
    evidence_id: uuid.UUID
    text: str
    claim_type: str
    gap_tier: str | None


@dataclass
class _ContributionClaimRow:
    paper_id: uuid.UUID
    text: str
    claim_type: str


def _load_gap_claims(session: Session, paper_ids: list[uuid.UUID]) -> list[_GapClaimRow]:
    rows = session.execute(
        select(
            ExtractedClaim.paper_id,
            ExtractedClaim.evidence_id,
            ExtractedClaim.text,
            ExtractedClaim.claim_type,
            ExtractedClaim.validation_tier,
        ).where(ExtractedClaim.paper_id.in_(paper_ids), ExtractedClaim.claim_type.in_(RELEVANT_CLAIM_TYPES))
    ).all()
    return [
        _GapClaimRow(paper_id=paper_id, evidence_id=evidence_id, text=text, claim_type=claim_type, gap_tier=tier)
        for paper_id, evidence_id, text, claim_type, tier in rows
    ]


def _load_contribution_claims(session: Session, paper_ids: list[uuid.UUID]) -> list[_ContributionClaimRow]:
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.claim_type).where(
            ExtractedClaim.paper_id.in_(paper_ids), ExtractedClaim.claim_type.in_(CONTRIBUTION_CLAIM_TYPES)
        )
    ).all()
    return [_ContributionClaimRow(paper_id=paper_id, text=text, claim_type=claim_type) for paper_id, text, claim_type in rows]


def _own_contribution_overlaps(
    gap_rows: list[_GapClaimRow], contribution_rows: list[_ContributionClaimRow], embedder: Embedder
) -> dict[uuid.UUID, float]:
    """Max cosine similarity between each gap claim and that SAME paper's
    own main_contribution/results claims - 0.0 if the paper has none, and
    never compared against another paper's contribution (that's a separate,
    intentionally distinct check - see find_addressing_papers)."""
    if not gap_rows:
        return {}

    contribution_texts_by_paper: dict[uuid.UUID, list[str]] = {}
    for row in contribution_rows:
        contribution_texts_by_paper.setdefault(row.paper_id, []).append(row.text)

    all_contribution_texts = [row.text for row in contribution_rows]
    gap_texts = [row.text for row in gap_rows]
    vectors = embedder.embed_texts(gap_texts + all_contribution_texts)
    gap_vectors = vectors[: len(gap_texts)]
    contribution_vectors = vectors[len(gap_texts) :]

    # if the same contribution text string repeats verbatim across two different
    # papers, this dict collapses to one entry - harmless, since identical text
    # implies identical embedding anyway
    vector_by_contribution_text: dict[str, list[float]] = dict(zip(all_contribution_texts, contribution_vectors, strict=True))

    overlaps: dict[uuid.UUID, float] = {}
    for row, vector in zip(gap_rows, gap_vectors, strict=True):
        own_texts = contribution_texts_by_paper.get(row.paper_id, [])
        if not own_texts:
            overlaps[row.evidence_id] = 0.0
            continue
        own_vectors = [vector_by_contribution_text[t] for t in own_texts]
        overlaps[row.evidence_id] = max(cosine_similarity(vector, ov) for ov in own_vectors)

    return overlaps
