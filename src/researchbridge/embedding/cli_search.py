from __future__ import annotations

import argparse
import logging
import uuid

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.embedding.search import find_similar_to_paper, search_by_text


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Find similar papers by embedding similarity.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paper-id", type=str, help="Find papers similar to this paper's UUID")
    group.add_argument("--query", type=str, help="Find papers similar to this free-text query")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    engine = make_engine()
    session_factory = make_session_factory(engine)
    session = session_factory()

    try:
        if args.paper_id:
            results = find_similar_to_paper(session, uuid.UUID(args.paper_id), embedder.model_name, args.top_k)
        else:
            results = search_by_text(session, args.query, embedder, args.top_k)

        for paper, distance in results:
            print(f"{distance:.4f}  {paper.source}:{paper.source_id}  {paper.title}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
