from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.extraction.base import Extractor
from researchbridge.extraction.heuristic import HeuristicExtractor
from researchbridge.extraction.hybrid import HybridExtractor
from researchbridge.extraction.pipeline import ExtractionPipeline, reset_extraction_data
from researchbridge.extraction.semantic import SemanticExtractor
from researchbridge.extraction.stub import StubExtractor

EXTRACTOR_NAMES = ("stub", "heuristic", "semantic", "hybrid")


def _make_extractor(name: str) -> Extractor:
    if name == "stub":
        print(
            "WARNING: using StubExtractor - output is synthetic placeholder data "
            "(extraction_method='stub'), not real analysis. Never use it as benchmark ground truth."
        )
        return StubExtractor()
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "semantic":
        return SemanticExtractor(SentenceTransformerEmbedder())
    if name == "hybrid":
        return HybridExtractor(SentenceTransformerEmbedder())
    raise ValueError(f"unknown extractor {name!r}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Run an extraction pass over ingested papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of unprocessed papers to extract")
    parser.add_argument(
        "--extractor",
        choices=EXTRACTOR_NAMES,
        default="hybrid",
        help="Which Extractor to run (default: hybrid - see extraction/hybrid.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing extracted claims/evidence first, then re-extract the whole corpus",
    )
    args = parser.parse_args()

    extractor = _make_extractor(args.extractor)

    engine = make_engine()
    session_factory = make_session_factory(engine)

    if args.force:
        session = session_factory()
        try:
            deleted = reset_extraction_data(session)
        finally:
            session.close()
        print(f"--force: deleted {deleted} existing evidence rows (and their claims).")

    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit, force=args.force)
    print(f"Extraction run {run_id} finished.")


if __name__ == "__main__":
    main()
