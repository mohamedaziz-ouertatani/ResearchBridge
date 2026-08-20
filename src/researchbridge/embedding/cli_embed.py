from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.embedding.pipeline import EmbeddingPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Compute embeddings for ingested papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of unembedded papers to process")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = EmbeddingPipeline(embedder=embedder, session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit)
    print(f"Embedding run {run_id} finished.")


if __name__ == "__main__":
    main()
