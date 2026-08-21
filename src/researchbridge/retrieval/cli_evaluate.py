"""Evaluates the four Sec 27 baselines against a query set.

Two query sets are available:

- "self" (relevance.py): each benchmark paper's own research_question,
  judged against itself. Cheap (reuses existing annotation work) but
  saturates at a perfect 1.0 for every baseline - the query is generated
  from the paper it's judged against, so it always retains enough of that
  paper's own phrasing to be a near-unique fingerprint for it, regardless
  of retrieval method. See relevance.py's docstring for how this was
  diagnosed.

- "topical" (topical_queries.py): queries written independent of any
  single paper, judged against pooled candidates from all four baselines,
  with possibly-multiple relevant documents per query. This is what
  actually answers Sec 27's question - does semantic or hybrid retrieval
  outperform lexical retrieval - since "self" cannot.

Defaults to running both, since seeing "self" saturate and "topical" not
saturate side by side is itself part of the honest story here.
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
from researchbridge.retrieval.topical_queries import DEFAULT_QUERIES_PATH, build_topical_query_set

DEFAULT_K = 10
QUERY_SETS = ("self", "topical")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Evaluate TF-IDF, BM25, embedding, and hybrid retrieval.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--query-set", choices=[*QUERY_SETS, "both"], default="both")
    args = parser.parse_args()

    session = make_session_factory(make_engine())()

    try:
        embedder = SentenceTransformerEmbedder()

        for name in QUERY_SETS if args.query_set == "both" else (args.query_set,):
            judgments, skipped = (
                build_query_set(session, args.benchmark_dir)
                if name == "self"
                else build_topical_query_set(session, args.queries_file)
            )
            print(f"\n=== query set: {name} ===")
            if skipped:
                print(f"Skipped {len(skipped)} unresolved judgment(s): {skipped}")
            if not judgments:
                print("No usable queries - nothing to evaluate.")
                continue
            print(f"{len(judgments)} queries, k={args.k}\n")

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
