"""Selects the Phase 1 benchmark sample and writes per-paper annotation files.

Idempotent: an existing annotation file is never overwritten (it may
already have hand-written content), so re-running after the corpus grows
only ever adds files, never clobbers annotation work in progress.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from researchbridge.benchmark.annotation_template import annotation_filename, render_annotation_template
from researchbridge.benchmark.domains import DEFAULT_TARGETS
from researchbridge.benchmark.sampling import stratified_sample
from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("benchmark")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Select the stratified benchmark sample and write annotation templates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = make_engine()
    session_factory = make_session_factory(engine)
    session = session_factory()

    try:
        sample = stratified_sample(session, seed=args.seed)
        written, skipped = _write_annotation_files(sample, args.output_dir)
        _write_manifest(sample, args.output_dir)

        for domain, target in DEFAULT_TARGETS.items():
            found = len(sample.get(domain, []))
            if found < target:
                logger.warning("Domain %r: only %d/%d papers available in the corpus", domain, found, target)

        print(f"Wrote {written} new annotation template(s), {skipped} already existed, in {args.output_dir}/annotations/")
        print(f"Manifest: {args.output_dir}/sample_manifest.csv")
    finally:
        session.close()


def _write_annotation_files(sample: dict[str, list], output_dir: Path) -> tuple[int, int]:
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for domain, papers in sample.items():
        for paper in papers:
            path = annotations_dir / annotation_filename(paper)
            if path.exists():
                skipped += 1
                continue
            path.write_text(render_annotation_template(paper, domain), encoding="utf-8")
            written += 1
    return written, skipped


def _write_manifest(sample: dict[str, list], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "source", "source_id", "year", "title", "url", "annotation_file"])
        for domain, papers in sample.items():
            for paper in papers:
                year = paper.publication_date.year if paper.publication_date else ""
                writer.writerow(
                    [domain, paper.source, paper.source_id, year, paper.title, paper.url or "", annotation_filename(paper)]
                )


if __name__ == "__main__":
    main()
