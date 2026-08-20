from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.connectors.arxiv import ArxivConnector
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.ingestion.pipeline import IngestionPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Run an arXiv ingestion pass.")
    parser.add_argument("--search-query", default="cat:cs.LG OR cat:cs.AI", help="arXiv search_query syntax")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: run to exhaustion)")
    args = parser.parse_args()

    connector = ArxivConnector(search_query=args.search_query, page_size=args.page_size)
    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)
    run_id = pipeline.run(max_pages=args.max_pages)
    print(f"Ingestion run {run_id} finished.")


if __name__ == "__main__":
    main()
