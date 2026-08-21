"""Baseline 4 (Sec 27): hybrid lexical + semantic retrieval.

Combines two retrievers by Reciprocal Rank Fusion (Cormack et al. 2009):
score(doc) = sum over rankers of 1 / (k_rrf + rank_in_that_ranker).

RRF fuses RANKS rather than raw scores, which sidesteps a real problem: BM25
scores and cosine similarities live on different, incomparable scales, so
averaging them directly would let whichever one happens to have larger
numbers dominate. A document missing from a ranker's candidate pool simply
contributes 0 from that ranker, rather than needing a made-up default score.
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.retrieval.base import Retriever

DEFAULT_K_RRF = 60
DEFAULT_CANDIDATE_POOL = 50


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        k_rrf: int = DEFAULT_K_RRF,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.k_rrf = k_rrf
        self.candidate_pool = candidate_pool

    def fit(self, session: Session) -> None:
        self.lexical.fit(session)
        self.semantic.fit(session)

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        lexical_hits = self.lexical.search(query, self.candidate_pool)
        semantic_hits = self.semantic.search(query, self.candidate_pool)

        rrf_scores: dict[uuid.UUID, float] = defaultdict(float)
        papers_by_id: dict[uuid.UUID, Paper] = {}
        for hits in (lexical_hits, semantic_hits):
            for rank, (paper, _score) in enumerate(hits, start=1):
                rrf_scores[paper.id] += 1.0 / (self.k_rrf + rank)
                papers_by_id[paper.id] = paper

        ranked_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)
        return [(papers_by_id[pid], rrf_scores[pid]) for pid in ranked_ids[:top_k]]
