from __future__ import annotations

import argparse
import logging
import os

from researchbridge.config import load_config
from researchbridge.connectors.semantic_scholar import SemanticScholarConnector
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.ingestion.pipeline import IngestionPipeline

DEFAULT_QUERY = '"artificial intelligence" | "machine learning" | "computer science"'


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Run a Semantic Scholar ingestion pass.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Semantic Scholar bulk search boolean query")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: run to exhaustion)")
    args = parser.parse_args()

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    connector = SemanticScholarConnector(query=args.query, api_key=api_key)
    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)
    run_id = pipeline.run(max_pages=args.max_pages)
    print(f"Ingestion run {run_id} finished.")


if __name__ == "__main__":
    main()
