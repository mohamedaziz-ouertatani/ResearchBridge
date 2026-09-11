"""One-off backfill: retroactively add PaperCategory rows to CORE papers
ingested before connectors/core.py's fieldOfStudy fix (2026-09-11). The
category mapping previously read a "subjects" key that doesn't exist
anywhere in CORE's real v3 response (the field-shape docs 403'd during
initial development, so it was guessed) - every CORE paper's categories
came back empty regardless of what CORE actually reported, which is why
CORE, the largest single ingested source, had zero rows in
paper_categories.

A plain re-run of ingestion cannot fix this: pipeline.py's _deduplicate
skips any paper already present in `papers` before insert (checked live,
2026-09-11 - a 3-page re-run against the default query fetched 300
already-known records, inserted 0), so PaperCategory is never revisited
for an existing row. This script instead re-fetches each existing CORE
paper by its own source_id and applies the corrected mapping directly.

Uses CORE's `id:(a OR b OR ...)` search-query syntax to look up up to 100
existing papers per request (verified live to work, both at 50 and 100
ids) rather than one request per paper - the difference between ~14
minutes and ~24 hours at the connector's 2s throttle for ~43k papers.

Idempotent: skips any CORE paper that already has a paper_categories row
(source="core"), so an interrupted run can simply be re-run; already-
processed papers cost nothing but the one indexed query that excludes
them.

Usage:
    uv run python scripts/backfill_core_categories.py                # dry run
    uv run python scripts/backfill_core_categories.py --apply         # writes
    uv run python scripts/backfill_core_categories.py --apply --limit 200  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from researchbridge.config import load_config
from researchbridge.db.models import Paper, PaperCategory
from researchbridge.db.session import make_engine

CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works/"
BATCH_SIZE = 100
MIN_REQUEST_INTERVAL_SECONDS = 2.0  # matches connectors/core.py's throttle


def _fetch_field_of_study(source_ids: list[str], api_key: str) -> dict[str, str]:
    """{source_id: fieldOfStudy} for every id in the batch CORE returned a
    non-null fieldOfStudy for. An id CORE didn't return a field for (still
    null on re-check, or dropped from CORE's index since ingestion) is
    simply absent from the result - never invented."""
    query = "id:(" + " OR ".join(source_ids) + ")"
    response = requests.get(
        CORE_SEARCH_URL,
        params={"q": query, "limit": len(source_ids)},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return {str(r["id"]): r["fieldOfStudy"] for r in results if r.get("fieldOfStudy")}


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N candidate papers (for testing)")
    args = parser.parse_args()

    load_config()
    api_key = os.environ.get("CORE_API_KEY")
    if not api_key:
        raise SystemExit("CORE_API_KEY is required (see .env.example)")

    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        # LEFT JOIN + IS NULL rather than a NOT IN subquery on an
        # already-categorized id set: avoids materializing a huge Python
        # set/IN-list and stays correct whether that set is empty or not.
        candidates = session.execute(
            select(Paper.id, Paper.source_id)
            .outerjoin(
                PaperCategory,
                and_(PaperCategory.paper_id == Paper.id, PaperCategory.source == "core"),
            )
            .where(Paper.source == "core", PaperCategory.id.is_(None))
            .order_by(Paper.created_at)
        ).all()

        if args.limit:
            candidates = candidates[: args.limit]

        print(f"{len(candidates)} CORE papers with no category rows yet")
        if not candidates:
            return

        by_id = {str(source_id): paper_id for paper_id, source_id in candidates}
        batches = _chunks(list(by_id.keys()), BATCH_SIZE)

        found: dict[str, str] = {}  # source_id -> fieldOfStudy
        last_request_at: float | None = None

        for i, batch in enumerate(batches, start=1):
            if last_request_at is not None:
                remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            last_request_at = time.monotonic()

            found.update(_fetch_field_of_study(batch, api_key))

            if i % 10 == 0 or i == len(batches):
                print(f"  batch {i}/{len(batches)}: {len(found)} categorized so far")

        still_empty = len(by_id) - len(found)
        print(f"\n{len(found)} papers now have a fieldOfStudy value; {still_empty} still don't (null on CORE's side)")

        if not found:
            return

        backup_path = Path(f"scripts/backfills/core_categories_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                [
                    {"paper_id": str(by_id[source_id]), "source_id": source_id, "category": category}
                    for source_id, category in found.items()
                ],
                indent=2,
            )
        )
        print(f"logged {len(found)} rows to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for source_id, category in found.items():
            session.add(
                PaperCategory(
                    paper_id=by_id[source_id],
                    category=category,
                    confidence="high",  # matches every other source's ingestion-time confidence - see PaperCategory docstring
                    source="core",
                )
            )
        session.commit()
        print(f"inserted {len(found)} paper_categories rows")


if __name__ == "__main__":
    main()
