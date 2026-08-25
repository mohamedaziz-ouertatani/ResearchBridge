from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.embedding.pipeline import EmbeddingPipeline, reset_embedding_data


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Compute embeddings for ingested papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of unembedded papers to process")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing embeddings for this model first, then re-embed the whole corpus",
    )
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    engine = make_engine()
    session_factory = make_session_factory(engine)

    if args.force:
        session = session_factory()
        try:
            deleted = reset_embedding_data(session, embedder.model_name)
        finally:
            session.close()
        print(f"--force: deleted {deleted} existing embeddings for model {embedder.model_name!r}.")

    pipeline = EmbeddingPipeline(embedder=embedder, session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit, force=args.force)
    print(f"Embedding run {run_id} finished.")


if __name__ == "__main__":
    main()
