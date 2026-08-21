"""Retrieval metrics (Sec 27): Precision@K, Recall@K, nDCG@K, MRR.

Pure functions over ranked ID lists and a relevance-judgment set, so they
don't depend on how the ranking was produced or how judgments were sourced
- any Retriever's output and any {paper_id: is_relevant} judgment set works.

Relevance is binary here (Sec 24's benchmark doesn't define graded
relevance), which makes nDCG's gain term 1/0 rather than a finer scale -
still meaningful as a rank-quality measure, just not using nDCG's full
graded-relevance expressiveness.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence


def precision_at_k(ranked_ids: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_ids: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for pid in ranked_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(ranked_ids: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID], k: int) -> float:
    top_k = ranked_ids[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, pid in enumerate(top_k, start=1) if pid in relevant_ids)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: Sequence[uuid.UUID], relevant_ids: set[uuid.UUID]) -> float:
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0
