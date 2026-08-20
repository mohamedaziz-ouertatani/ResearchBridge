"""Downloads full text for every paper in the benchmark sample.

Resumable and idempotent: an already-cached paper is skipped, so a run
interrupted halfway costs only the papers it hadn't reached.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

from researchbridge.benchmark.fulltext import fetch_fulltext, fulltext_path, throttle
from researchbridge.benchmark.store import load_all
from researchbridge.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_DIR = Path("benchmark")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Fetch full text for the benchmark papers.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    args = parser.parse_args()

    output_dir = args.benchmark_dir / "fulltext"
    annotations = load_all(args.benchmark_dir)
    if not annotations:
        print(f"No annotation files in {args.benchmark_dir}/annotations/ - run rb-benchmark-sample first.")
        return

    fetched = skipped = 0
    failures: list[tuple[str, str]] = []
    http = requests.Session()

    for i, annotation in enumerate(annotations, start=1):
        source_id = annotation.source_id
        if fulltext_path(output_dir, source_id).exists():
            skipped += 1
            continue

        if fetched > 0:
            throttle()

        try:
            text = fetch_fulltext(source_id, output_dir, session=http)
        except Exception as exc:  # one unavailable PDF must not end the run
            failures.append((source_id, str(exc)[:200]))
            print(f"[{i}/{len(annotations)}] {source_id}: FAILED {exc}", flush=True)
            continue

        fetched += 1
        print(f"[{i}/{len(annotations)}] {source_id}: {len(text.split()):,} words", flush=True)

    print(f"\nfetched {fetched}, already cached {skipped}, failed {len(failures)}")
    for source_id, detail in failures:
        print(f"  {source_id}: {detail}")


if __name__ == "__main__":
    main()
