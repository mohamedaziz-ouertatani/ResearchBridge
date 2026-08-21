"""Evaluates an Extractor against the Sec 25 benchmark, per Sec 28's rules:
report each field's Precision/Recall/F1 independently, never one overall
number.

Runs the extractor directly against each benchmark paper's abstract (not
through the persistence pipeline - this is measuring the extractor, not
exercising storage) and compares each field's extracted text to the
matching ground-truth field via embedding similarity (evaluation.py).

Scope: research_gap is not evaluated here - it's a nested {addressed,
remaining} pair in the benchmark schema, not a flat text field like the
other eight, and forcing it into the same comparison would be misleading
rather than useful. A fair evaluation for it needs its own design.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import select

from researchbridge.benchmark.cli_sample import DEFAULT_OUTPUT_DIR
from researchbridge.benchmark.store import ANNOTATION_FIELDS, load_all
from researchbridge.config import load_config
from researchbridge.db.models import Paper
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.extraction.base import Extractor
from researchbridge.extraction.evaluation import DEFAULT_SIMILARITY_THRESHOLD, evaluate
from researchbridge.extraction.heuristic import HeuristicExtractor


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Evaluate an extractor against the Sec 25 benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    session = make_session_factory(make_engine())()

    try:
        annotations = load_all(args.benchmark_dir)
        if not annotations:
            print(f"No annotations in {args.benchmark_dir}/annotations/ - run rb-benchmark-sample first.")
            return

        source_ids = [a.source_id for a in annotations]
        papers_by_source_id = {
            p.source_id: p
            for p in session.execute(
                select(Paper).where(Paper.source == "arxiv", Paper.source_id.in_(source_ids))
            ).scalars()
        }

        extractor: Extractor = HeuristicExtractor()
        predictions, ground_truth, skipped = {}, {}, []
        for annotation in annotations:
            paper = papers_by_source_id.get(annotation.source_id)
            if paper is None:
                skipped.append(annotation.source_id)
                continue
            predictions[annotation.source_id] = extractor.extract(paper)
            ground_truth[annotation.source_id] = annotation.fields

        if skipped:
            print(f"Skipped {len(skipped)} benchmark paper(s) missing from the corpus: {skipped}")
        print(f"Evaluating {extractor.extraction_method!r} on {len(ground_truth)} papers, threshold={args.threshold}\n")

        embedder = SentenceTransformerEmbedder()
        scores = evaluate(predictions, ground_truth, list(ANNOTATION_FIELDS), embedder, args.threshold)
        _print_table(scores)
    finally:
        session.close()


def _print_table(scores) -> None:
    header = f"{'field':<20}{'precision':>12}{'recall':>10}{'f1':>10}"
    print(header)
    print("-" * len(header))
    for field, s in scores.items():
        print(f"{field:<20}{s.precision:>12.3f}{s.recall:>10.3f}{s.f1:>10.3f}")


if __name__ == "__main__":
    main()
