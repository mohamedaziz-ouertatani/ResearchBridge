"""Reports whether analysis_claims confidence buckets correlate with real
human review ratings (see gaps/claims_calibration.py). Read-only - never
writes to candidate_gaps or analysis_claims, only to its own results file.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.gaps.claims_calibration import gap_claim_confidence_vs_ratings

DEFAULT_RESULTS_PATH = Path("benchmark/claims_calibration_results.json")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(
        description="Check analysis_claims confidence buckets against human review ratings."
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()

    session = make_session_factory(make_engine())()
    try:
        stats = gap_claim_confidence_vs_ratings(session)
    finally:
        session.close()

    print("Gap-derived claim confidence vs. human review ratings:")
    if len(stats) <= 1:
        print(
            "  Only one confidence bucket present in reviewed data - no cross-bucket "
            "comparison is possible yet (see this module's docstring)."
        )
    for bucket in stats:
        print(f"  confidence={bucket.confidence}: {bucket.sample_count} reviewed gap(s)")
        for field, mean in bucket.mean_ratings.items():
            print(f"    {field}: mean {mean:.2f}")

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "buckets": [
            {"confidence": bucket.confidence, "sample_count": bucket.sample_count, "mean_ratings": bucket.mean_ratings}
            for bucket in stats
        ],
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output_file}")


if __name__ == "__main__":
    main()
