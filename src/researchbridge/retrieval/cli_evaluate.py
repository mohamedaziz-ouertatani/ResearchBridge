"""Evaluates the four Sec 27 baselines against the benchmark-derived query set.

Answers Sec 27's research question - does semantic or hybrid retrieval
outperform simple lexical retrieval for scientific-paper discovery - with
actual numbers, not just eyeballed rankings (that's what rb-retrieval-compare
is for).
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

from researchbridge.benchmark.cli_sample import DEFAULT_OUTPUT_DIR
from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.retrieval.base import Retriever
from researchbridge.retrieval.bm25 import Bm25Retriever
from researchbridge.retrieval.embedding_retriever import EmbeddingRetriever
from researchbridge.retrieval.hybrid import HybridRetriever
from researchbridge.retrieval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from researchbridge.retrieval.relevance import QueryJudgment, build_query_set
from researchbridge.retrieval.tfidf import TfidfRetriever

DEFAULT_K = 10


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF, BM25, embedding, and hybrid retrieval against the benchmark query set."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    session = make_session_factory(make_engine())()

    try:
        judgments, skipped = build_query_set(session, args.benchmark_dir)
        if skipped:
            print(f"Skipped {len(skipped)} benchmark paper(s) with no usable query or no corpus match: {skipped}")
        if not judgments:
            print("No usable queries - nothing to evaluate.")
            return
        print(f"Evaluating on {len(judgments)} queries (one per benchmark paper), k={args.k}\n")

        embedder = SentenceTransformerEmbedder()
        retrievers: list[Retriever] = [
            TfidfRetriever(),
            Bm25Retriever(),
            EmbeddingRetriever(embedder),
            HybridRetriever(lexical=Bm25Retriever(), semantic=EmbeddingRetriever(embedder)),
        ]

        rows = []
        for retriever in retrievers:
            retriever.fit(session)
            rows.append((retriever.name, _evaluate(retriever, judgments, args.k)))

        _print_table(rows, args.k)
    finally:
        session.close()


def _evaluate(retriever: Retriever, judgments: list[QueryJudgment], k: int) -> dict[str, float]:
    totals = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    for judgment in judgments:
        hits = retriever.search(judgment.query, top_k=max(k, 20))
        ranked_ids: list[uuid.UUID] = [paper.id for paper, _score in hits]

        totals["precision"] += precision_at_k(ranked_ids, judgment.relevant_ids, k)
        totals["recall"] += recall_at_k(ranked_ids, judgment.relevant_ids, k)
        totals["ndcg"] += ndcg_at_k(ranked_ids, judgment.relevant_ids, k)
        totals["mrr"] += mrr(ranked_ids, judgment.relevant_ids)

    n = len(judgments)
    return {metric: total / n for metric, total in totals.items()}


def _print_table(rows: list[tuple[str, dict[str, float]]], k: int) -> None:
    header = f"{'method':<12}{f'P@{k}':>10}{f'R@{k}':>10}{f'nDCG@{k}':>10}{'MRR':>10}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['ndcg']:>10.3f}{m['mrr']:>10.3f}")


if __name__ == "__main__":
    main()
