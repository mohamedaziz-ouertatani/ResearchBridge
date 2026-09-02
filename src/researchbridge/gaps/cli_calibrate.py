"""Runs gaps/calibration.py against the real corpus + real embedder,
mirroring extraction/cli_evaluate.py's structure: load ground truth, run
the measurement, print a table, write a JSON results file for the admin
dashboard / future re-runs.

Runs OWN_CONTRIBUTION_OVERLAP_THRESHOLD and ADDRESSING_SIMILARITY_THRESHOLD
as two separate sweeps over two separate sample pools, never combined -
they measure different things (own-paper overlap vs. cross-paper warning-
worthiness) and are calibrated independently (see calibration.py's module
docstring and Task 14).

Dry run, read-only: never writes to candidate_gaps or any pipeline table,
only to benchmark/gap_calibration_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from researchbridge.benchmark.cli_sample import DEFAULT_OUTPUT_DIR
from researchbridge.benchmark.store import load_all
from researchbridge.config import load_config
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.gaps.calibration import (
    addressing_signal_samples,
    load_calibration_groups,
    own_contribution_overlap_group_samples,
    own_contribution_overlap_samples,
    sweep_thresholds,
)

DEFAULT_GROUPS_PATH = Path("benchmark/gap_calibration_groups.yaml")
DEFAULT_RESULTS_PATH = Path("benchmark/gap_calibration_results.json")
DEFAULT_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Calibrate gap-signal overlap thresholds against real annotations.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--groups-file", type=Path, default=DEFAULT_GROUPS_PATH)
    parser.add_argument("--thresholds", type=str, default=None, help="comma-separated, e.g. 0.3,0.4,0.5")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    annotations = load_all(args.benchmark_dir)
    if not annotations:
        print(f"No annotations in {args.benchmark_dir}/annotations/ - run rb-benchmark-sample first.")
        return

    groups = load_calibration_groups(args.groups_file)
    thresholds = (
        [float(t) for t in args.thresholds.split(",")] if args.thresholds else DEFAULT_THRESHOLDS
    )

    embedder = SentenceTransformerEmbedder()

    own_overlap_samples = own_contribution_overlap_samples(annotations, embedder) + own_contribution_overlap_group_samples(
        annotations, groups, embedder
    )
    addressing_samples = addressing_signal_samples(annotations, groups, embedder)

    own_overlap_rows = sweep_thresholds(own_overlap_samples, thresholds) if own_overlap_samples else []
    addressing_rows = sweep_thresholds(addressing_samples, thresholds) if addressing_samples else []

    if not own_overlap_rows and not addressing_rows:
        print("No calibration samples available - check benchmark annotations and gap_calibration_groups.yaml.")
        return

    print("=== OWN_CONTRIBUTION_OVERLAP_THRESHOLD calibration (same-paper overlap) ===")
    if own_overlap_rows:
        _print_table(own_overlap_rows)
    else:
        print("(no samples - check gap_calibration_groups.yaml's recurring_limitation_* groups)")

    print("\n=== ADDRESSING_SIMILARITY_THRESHOLD calibration (cross-paper warning signal) ===")
    if addressing_rows:
        _print_table(addressing_rows)
    else:
        print("(no samples - check gap_calibration_groups.yaml's high_value_addressing_match/"
              "false_warning_topical_only/benchmark_evaluation_partial_solution groups)")

    _write_results_json(
        args.output_file,
        own_overlap_rows=own_overlap_rows,
        addressing_rows=addressing_rows,
        own_overlap_sample_count=len(own_overlap_samples),
        addressing_sample_count=len(addressing_samples),
    )
    print(f"\nWrote results to {args.output_file}")


def _print_table(rows) -> None:
    header = f"{'threshold':>10}{'category':<38}{'n':>6}{'false_pos_rate':>16}{'false_neg_rate':>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.threshold:>10.2f}{row.category:<38}{row.total:>6}"
            f"{row.false_positive_rate:>16.3f}{row.false_negative_rate:>16.3f}"
        )


def _rows_to_json(rows) -> list[dict]:
    return [
        {
            "threshold": r.threshold,
            "category": r.category,
            "total": r.total,
            "false_positive_rate": r.false_positive_rate,
            "false_negative_rate": r.false_negative_rate,
        }
        for r in rows
    ]


def _write_results_json(
    path: Path, own_overlap_rows, addressing_rows, own_overlap_sample_count: int, addressing_sample_count: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "own_contribution_overlap": {
            "sample_count": own_overlap_sample_count,
            "rows": _rows_to_json(own_overlap_rows),
        },
        "addressing_similarity": {
            "sample_count": addressing_sample_count,
            "rows": _rows_to_json(addressing_rows),
        },
    }
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
