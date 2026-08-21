"""Detects candidate research gaps (Sec 32) for one seed paper's neighborhood,
or for every embedded paper in the corpus with --all.

Dry run by default: prints what it would produce. --save persists as
status="pending" CandidateGap rows for later human review - never
auto-approved (Sec 35/44).
"""

from __future__ import annotations

import argparse
import logging

from sqlalchemy import select

from researchbridge.config import load_config
from researchbridge.db.models import Paper
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.gaps.batch import run_all
from researchbridge.gaps.cluster import DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_SIMILARITY_THRESHOLD
from researchbridge.gaps.detect import detect_candidate_gaps
from researchbridge.gaps.persistence import save_candidate_gaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect recurring cross-paper gap patterns.")
    parser.add_argument("source_id", type=str, nargs="?", default=None, help="arXiv id of the seed paper (omit with --all)")
    parser.add_argument("--all", action="store_true", help="run over every embedded paper without an existing candidate gap")
    parser.add_argument("--force", action="store_true", help="with --all, reprocess papers that already have a candidate gap")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--save", action="store_true", help="persist as pending CandidateGap rows (default: dry run)")
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
        embedder = SentenceTransformerEmbedder()

        if args.all:
            _run_batch(session, embedder, args)
        else:
            _run_single(session, embedder, args)
    finally:
        session.close()


def _run_single(session, embedder, args: argparse.Namespace) -> None:
    paper = session.execute(
        select(Paper).where(Paper.source == "arxiv", Paper.source_id == args.source_id)
    ).scalar_one_or_none()
    if paper is None:
        print(f"No paper with arxiv id {args.source_id} in the corpus.")
        return

    try:
        drafts = detect_candidate_gaps(
            session, paper.id, embedder, args.top_k, args.min_cluster_size, args.threshold
        )
    except ValueError as exc:
        # find_similar_to_paper raises if the seed has no embedding - a
        # real, nameable state, not a crash
        print(str(exc))
        return

    print(f"Seed: {paper.title}")
    print(
        f"Neighborhood: top {args.top_k} related papers, "
        f"min cluster size {args.min_cluster_size}, threshold {args.threshold}\n"
    )

    if not drafts:
        print("No recurring pattern found - not a failure, just nothing cleared the threshold.")
        return

    for i, draft in enumerate(drafts, start=1):
        print(f"[{i}] {draft.observation}")
        print(f"    contributing papers: {draft.contributing_paper_count}\n")

    if args.save:
        saved = save_candidate_gaps(session, drafts, args.threshold)
        print(f"Saved {len(saved)} candidate gap(s) with status='pending'.")


def _run_batch(session, embedder, args: argparse.Namespace) -> None:
    mode = "saving" if args.save else "dry run"
    print(f"Running gap detection over every embedded paper ({mode})...")

    summary = run_all(
        session,
        embedder,
        top_k=args.top_k,
        min_cluster_size=args.min_cluster_size,
        similarity_threshold=args.threshold,
        force=args.force,
        save=args.save,
    )

    print(f"Papers seen:    {summary.papers_seen}")
    print(f"Papers skipped: {summary.papers_skipped} (already had a candidate gap)")
    print(f"Papers failed:  {summary.papers_failed}")
    print(f"Gaps found:     {summary.gaps_found}")
    print(f"Gaps saved:     {summary.gaps_saved}")


if __name__ == "__main__":
    main()
