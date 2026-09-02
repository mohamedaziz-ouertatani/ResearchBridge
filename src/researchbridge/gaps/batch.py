"""Batch gap detection over every embedded paper in the corpus (Sec 32).

Reuses one Embedder instance across the whole run and commits after each
paper - mirrors ExtractionPipeline's per-paper commit/error-isolation
pattern (extraction/pipeline.py), so one bad paper never aborts the run
and a crash partway through loses at most the paper in progress.

Idempotent by default: a paper already present as a CandidateGap.seed_paper_id
is skipped on the next run (--force reprocesses it). No new tracking table -
candidate_gaps.seed_paper_id already answers "was this paper done", and this
project's own principle is to not add schema until complexity forces it.

Deliberately does not dedupe near-duplicate observations across seeds whose
neighborhoods overlap - a human reviewer rejecting a duplicate in the review
UI is safer than an unproven similarity heuristic silently merging two
observations that turn out to be genuinely distinct.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, Embedding, Paper
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.gaps.cluster import DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_SIMILARITY_THRESHOLD
from researchbridge.gaps.detect import detect_candidate_gaps
from researchbridge.gaps.persistence import save_candidate_gaps

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    papers_seen: int = 0
    papers_skipped: int = 0
    papers_failed: int = 0
    no_relevant_papers: int = 0
    insufficient_evidence: int = 0
    gaps_found: int = 0
    gaps_saved: int = 0


def run_all(
    session: Session,
    embedder: Embedder,
    top_k: int = 15,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    force: bool = False,
    save: bool = False,
) -> BatchSummary:
    summary = BatchSummary()
    seed_ids = _select_seed_papers(session, force)
    total_embedded = _count_embedded(session)
    summary.papers_skipped = total_embedded - len(seed_ids)

    for paper_id in seed_ids:
        summary.papers_seen += 1
        try:
            result = detect_candidate_gaps(session, paper_id, embedder, top_k, min_cluster_size, similarity_threshold)
        except Exception:
            logger.exception("Gap detection failed for paper %s", paper_id)
            summary.papers_failed += 1
            continue

        if result.status == "no_relevant_papers":
            summary.no_relevant_papers += 1
        elif result.status == "insufficient_evidence":
            summary.insufficient_evidence += 1

        summary.gaps_found += len(result.drafts)
        if result.drafts and save:
            saved = save_candidate_gaps(session, result.drafts, similarity_threshold)
            summary.gaps_saved += len(saved)

    return summary


def _select_seed_papers(session: Session, force: bool) -> list[uuid.UUID]:
    query = select(Paper.id).where(
        exists().where(Embedding.paper_id == Paper.id, Embedding.embedding_type == EMBEDDING_TYPE),
        Paper.excluded_at.is_(None),
    )
    if not force:
        already_done = exists().where(CandidateGap.seed_paper_id == Paper.id)
        query = query.where(~already_done)
    return list(session.execute(query).scalars())


def _count_embedded(session: Session) -> int:
    query = select(Paper.id).where(
        exists().where(Embedding.paper_id == Paper.id, Embedding.embedding_type == EMBEDDING_TYPE),
        Paper.excluded_at.is_(None),
    )
    return len(list(session.execute(query).scalars()))
