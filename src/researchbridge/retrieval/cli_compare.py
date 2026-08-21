"""Runs one query against all four Sec 27 baselines side by side.

For eyeballing retrieval quality on the real corpus without needing the
formal relevance-judgment set the evaluation harness will require - that's
a separate, not-yet-made methodology decision (see Sec 27: "use manually
judged relevance labels").
"""

from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.retrieval.bm25 import Bm25Retriever
from researchbridge.retrieval.embedding_retriever import EmbeddingRetriever
from researchbridge.retrieval.hybrid import HybridRetriever
from researchbridge.retrieval.tfidf import TfidfRetriever


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Compare TF-IDF, BM25, embedding, and hybrid retrieval on one query.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    engine = make_engine()
    session = make_session_factory(engine)()

    try:
        embedder = SentenceTransformerEmbedder()

        # each retriever gets its own instance and calls fit() itself below
        # (including the two HybridRetriever wraps internally), so nothing
        # here is shared or fit twice against the caller's intent.
        retrievers = [
            TfidfRetriever(),
            Bm25Retriever(),
            EmbeddingRetriever(embedder),
            HybridRetriever(lexical=Bm25Retriever(), semantic=EmbeddingRetriever(embedder)),
        ]

        for retriever in retrievers:
            print(f"\n=== {retriever.name} ===")
            retriever.fit(session)
            hits = retriever.search(args.query, args.top_k)
            if not hits:
                print("  (no results)")
            for paper, score in hits:
                print(f"  {score:.4f}  {paper.source}:{paper.source_id}  {paper.title}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
