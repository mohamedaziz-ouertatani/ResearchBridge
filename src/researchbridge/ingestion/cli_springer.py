from __future__ import annotations

import argparse
import logging
import os

from researchbridge.config import load_config
from researchbridge.connectors.springer import SpringerConnector
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.ingestion.pipeline import IngestionPipeline

DEFAULT_QUERY = '"artificial intelligence" OR "machine learning" OR "computer science"'


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Run a Springer Nature ingestion pass.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Springer Nature Meta API free-text query")
    parser.add_argument("--page-size", type=int, default=25)  # free-tier ceiling, verified live
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: run to exhaustion)")
    args = parser.parse_args()

    api_key = os.environ.get("SPRINGER_META_API_KEY")
    connector = SpringerConnector(query=args.query, api_key=api_key, page_size=args.page_size)
    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)
    run_id = pipeline.run(max_pages=args.max_pages)
    print(f"Ingestion run {run_id} finished.")


if __name__ == "__main__":
    main()
