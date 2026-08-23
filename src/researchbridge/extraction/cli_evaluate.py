"""Evaluates Extractors against the Sec 25 benchmark, per Sec 28's rules:
report each field's Precision/Recall/F1 independently, never one overall
number.

Runs both baselines by default (heuristic, then the semantic improvement)
so the two tables sit side by side - the point of an Improvement step is
showing whether it actually improved things, field by field, not just
that a new method exists.

Each extractor runs directly against the benchmark papers' abstracts (not
through the persistence pipeline - this is measuring the extractor, not
exercising storage) and compares each field's extracted text to the
matching ground-truth field via embedding similarity (evaluation.py).

research_gap is evaluated against the benchmark's research_gap.remaining
specifically (Sec 32's "explicit gap": what the paper says is still open),
not .addressed - .addressed overlaps heavily with problem/main_contribution,
which are already evaluated separately, and Phase 3's actual use for this
field is finding what's NOT yet solved.
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
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.extraction.base import Extractor
from researchbridge.extraction.evaluation import DEFAULT_SIMILARITY_THRESHOLD, FieldScore, evaluate
from researchbridge.extraction.heuristic import HeuristicExtractor
from researchbridge.extraction.hybrid import HybridExtractor
from researchbridge.extraction.semantic import SemanticExtractor
from researchbridge.extraction.type_validation_evaluation import TypeValidationScore, evaluate_claim_type_validation

EXTRACTORS = ("heuristic", "semantic", "hybrid")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Evaluate one or both extractors against the Sec 25 benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--extractor", choices=[*EXTRACTORS, "all"], default="all")
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

        skipped = [a.source_id for a in annotations if a.source_id not in papers_by_source_id]
        usable = [a for a in annotations if a.source_id in papers_by_source_id]
        if skipped:
            print(f"Skipped {len(skipped)} benchmark paper(s) missing from the corpus: {skipped}")
        if not usable:
            print("No usable benchmark papers - nothing to evaluate.")
            return

        embedder = SentenceTransformerEmbedder()
        names = EXTRACTORS if args.extractor == "all" else (args.extractor,)
        for name in names:
            extractor = _make_extractor(name, embedder)
            scores = _evaluate_one(extractor, usable, papers_by_source_id, embedder, args.threshold)
            print(f"\n=== {extractor.extraction_method} ({len(usable)} papers, threshold={args.threshold}) ===")
            _print_table(scores)

        print(f"\n=== claim-type validation ({len(usable)} papers, ground-truth only) ===")
        _print_type_validation_table(evaluate_claim_type_validation(usable))
    finally:
        session.close()


def _make_extractor(name: str, embedder: Embedder) -> Extractor:
    if name == "heuristic":
        return HeuristicExtractor()
    if name == "semantic":
        return SemanticExtractor(embedder)
    if name == "hybrid":
        return HybridExtractor(embedder)
    raise ValueError(f"unknown extractor {name!r}")


def _evaluate_one(extractor, annotations, papers_by_source_id, embedder: Embedder, threshold: float) -> dict[str, FieldScore]:
    predictions = {a.source_id: extractor.extract(papers_by_source_id[a.source_id]) for a in annotations}
    ground_truth = {
        a.source_id: {**a.fields, "research_gap": a.research_gap.get("remaining", "")} for a in annotations
    }
    fields = [*ANNOTATION_FIELDS, "research_gap"]
    return evaluate(predictions, ground_truth, fields, embedder, threshold)


def _print_table(scores: dict[str, FieldScore]) -> None:
    header = f"{'field':<20}{'precision':>12}{'recall':>10}{'f1':>10}"
    print(header)
    print("-" * len(header))
    for field, s in scores.items():
        print(f"{field:<20}{s.precision:>12.3f}{s.recall:>10.3f}{s.f1:>10.3f}")


def _print_type_validation_table(scores: dict[str, TypeValidationScore]) -> None:
    header = f"{'field':<20}{'own-field accept':>18}{'cross-field reject':>20}"
    print(header)
    print("-" * len(header))
    for field, s in scores.items():
        own = f"{s.own_field_acceptance_rate:.3f} ({s.own_field_accepted}/{s.own_field_total})"
        cross = f"{s.cross_field_rejection_rate:.3f} ({s.cross_field_rejected}/{s.cross_field_total})"
        print(f"{field:<20}{own:>18}{cross:>20}")


if __name__ == "__main__":
    main()
