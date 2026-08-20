from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.extraction.pipeline import ExtractionPipeline
from researchbridge.extraction.stub import StubExtractor


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Run an extraction pass over ingested papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of unprocessed papers to extract")
    args = parser.parse_args()

    extractor = StubExtractor()
    print(
        "WARNING: using StubExtractor - output is synthetic placeholder data "
        "(extraction_method='stub'), not real analysis. Never use it as benchmark ground truth."
    )

    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit)
    print(f"Extraction run {run_id} finished.")


if __name__ == "__main__":
    main()
