from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.fulltext.pipeline import FullTextFetchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and parse full text for open-access papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of papers to process")
    parser.add_argument("--force", action="store_true", help="Re-fetch papers that already have full text")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = build_parser()
    args = parser.parse_args()

    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit, force=args.force)
    print(f"Full-text fetch run {run_id} finished.")


if __name__ == "__main__":
    main()
