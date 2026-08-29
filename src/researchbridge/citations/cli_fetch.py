"""Fetches citation edges for one paper, or every eligible paper with --all,
from either Semantic Scholar or CrossRef (--source). Dry run by default:
prints what it would create. --save persists PaperCitation rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from researchbridge.citations.batch import BatchSummary, run_all
from researchbridge.citations.fetch import (
    CrossrefCitationFetcher,
    SemanticScholarCitationFetcher,
    persist_citation_edges,
    resolve_citations,
)
from researchbridge.config import load_config
from researchbridge.db.models import CitationFetchRun, Paper
from researchbridge.db.session import make_engine, make_session_factory

SUMMARY_PATH_BY_SOURCE: dict[str, Path] = {
    "semantic_scholar": Path("benchmark/citations_fetch_summary_semantic_scholar.json"),
    "crossref": Path("benchmark/citations_fetch_summary_crossref.json"),
}


def summary_path_for(source: str) -> Path:
    """Each source gets its own persisted summary file - a Semantic Scholar
    run and a CrossRef run must not clobber each other's last-run stats,
    since they're independent passes with independent coverage."""
    return SUMMARY_PATH_BY_SOURCE[source]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch citation edges from Semantic Scholar or CrossRef.")
    parser.add_argument(
        "source_id",
        type=str,
        nargs="?",
        default=None,
        help="Semantic Scholar paperId, or CrossRef DOI when --source crossref (omit with --all)",
    )
    parser.add_argument(
        "--source",
        choices=["semantic_scholar", "crossref"],
        default="semantic_scholar",
        help="citation source to fetch from (default: semantic_scholar)",
    )
    parser.add_argument("--all", action="store_true", help="run over every eligible paper without existing outgoing edges from this source")
    parser.add_argument("--force", action="store_true", help="with --all, reprocess papers that already have outgoing edges")
    parser.add_argument("--save", action="store_true", help="persist PaperCitation rows (default: dry run)")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    parser = parser or build_parser()
    if args.all and args.source_id:
        parser.error("pass either a source_id or --all, not both")
    if not args.all and not args.source_id:
        parser.error("source_id is required unless --all is given")
    if args.force and not args.all:
        parser.error("--force only applies with --all")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    session = make_session_factory(make_engine())()
    try:
        if args.source == "crossref":
            fetcher = CrossrefCitationFetcher(mailto=os.environ.get("CROSSREF_MAILTO"))
        else:
            fetcher = SemanticScholarCitationFetcher(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))

        if args.all:
            _run_batch(session, fetcher, args)
        else:
            _run_single(session, fetcher, args)
    finally:
        session.close()


def _run_single(session, fetcher, args: argparse.Namespace) -> None:
    if args.source == "crossref":
        paper = session.execute(select(Paper).where(Paper.doi == args.source_id)).scalar_one_or_none()
    else:
        paper = session.execute(
            select(Paper).where(Paper.source == "semantic_scholar", Paper.source_id == args.source_id)
        ).scalar_one_or_none()
    if paper is None:
        print(f"No paper with id {args.source_id} in the corpus for source {args.source}.")
        return

    payload = fetcher.fetch_raw(args.source_id)
    resolution = resolve_citations(session, paper, payload, source=args.source)

    print(f"Seed: {paper.title}")
    print(f"Resolved edges: {len(resolution.edges)}")
    print(f"Unresolved citing (outside corpus): {resolution.unresolved_citing}")
    print(f"Unresolved cited (outside corpus): {resolution.unresolved_cited}")

    if args.save:
        created = persist_citation_edges(session, resolution, source=args.source)
        session.commit()
        print(f"Saved {created} new citation edge(s).")


def _run_batch(session, fetcher, args: argparse.Namespace) -> None:
    mode = "saving" if args.save else "dry run"
    print(f"Running citation fetch over every {args.source} paper ({mode})...")

    run = CitationFetchRun(source=args.source, status="running")
    session.add(run)
    session.commit()

    try:
        summary = run_all(session, fetcher, source=args.source, force=args.force, save=args.save, run=run)
    except Exception as exc:
        run.status = "failed"
        run.error_summary = str(exc)[:2000]
        run.finished_at = datetime.now(UTC)
        session.commit()
        raise

    # run's counts are already kept current by run_all (see batch.py's
    # _sync_run) - only the terminal status/finished_at need setting here.
    run.status = "completed"
    run.finished_at = datetime.now(UTC)
    session.commit()

    print(f"Papers seen:            {summary.papers_seen}")
    print(f"Papers failed:          {summary.papers_failed}")
    print(f"Edges created:          {summary.edges_created}")
    print(f"Edges already existed:  {summary.edges_already_existed}")

    summary_path = summary_path_for(args.source)
    write_summary_json(summary_path, summary)
    print(f"\nWrote summary to {summary_path}")


def _build_summary_json(summary: BatchSummary) -> dict[str, Any]:
    """Shapes a BatchSummary into the JSON the admin dashboard reads
    (GET /api/admin/citations-fetch) - a flat file, not a DB table, since
    this is a background job over the corpus, not a repeating pipeline
    stage (same "no run-history table" choice as retrieval evaluation)."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "papers_seen": summary.papers_seen,
        "papers_failed": summary.papers_failed,
        "edges_created": summary.edges_created,
        "edges_already_existed": summary.edges_already_existed,
    }


def write_summary_json(path: Path, summary: BatchSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_build_summary_json(summary), indent=2))


if __name__ == "__main__":
    main()
