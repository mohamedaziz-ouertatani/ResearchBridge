from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.fulltext.chunk_pipeline import FullTextChunkPipeline, reset_chunk_data
from researchbridge.pipeline_logging import configure_pipeline_logging


def main() -> None:
    configure_pipeline_logging("fulltext-chunk", logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Chunk and embed full-text paragraphs for open-access papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of papers to chunk")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing chunks for this model first, then re-chunk every paper with full text",
    )
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    engine = make_engine()
    session_factory = make_session_factory(engine)

    if args.force:
        session = session_factory()
        try:
            deleted = reset_chunk_data(session, embedder.model_name)
        finally:
            session.close()
        print(f"--force: deleted {deleted} existing chunks for model {embedder.model_name!r}.")

    pipeline = FullTextChunkPipeline(embedder=embedder, session_factory=session_factory)
    result = pipeline.run(limit=args.limit)
    print(
        f"Full-text chunk run finished: {result['papers_processed']} papers processed, "
        f"{result['chunks_created']} chunks created."
    )


if __name__ == "__main__":
    main()
