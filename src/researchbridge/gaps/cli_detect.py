"""Detects candidate research gaps (Sec 32) for one seed paper's neighborhood.

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
from researchbridge.gaps.cluster import DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_SIMILARITY_THRESHOLD
from researchbridge.gaps.detect import detect_candidate_gaps
from researchbridge.gaps.persistence import save_candidate_gaps


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    load_config()

    parser = argparse.ArgumentParser(description="Detect recurring cross-paper gap patterns for one paper's neighborhood.")
    parser.add_argument("source_id", type=str, help="arXiv id of the seed paper")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--save", action="store_true", help="persist as pending CandidateGap rows (default: dry run)")
    args = parser.parse_args()

    session = make_session_factory(make_engine())()
    try:
        paper = session.execute(
            select(Paper).where(Paper.source == "arxiv", Paper.source_id == args.source_id)
        ).scalar_one_or_none()
        if paper is None:
            print(f"No paper with arxiv id {args.source_id} in the corpus.")
            return

        embedder = SentenceTransformerEmbedder()
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
    finally:
        session.close()


if __name__ == "__main__":
    main()
