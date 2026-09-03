"""One-off backfill: re-applies connectors/base.py's clean_harvested_abstract()
(now with html.unescape() + stripped-HTML-entity repair, added 2026-09-03) to
every already-ingested paper's abstract, fixing corruption baked in before
the fix existed (e.g. CORE-sourced abstracts with "201C;"/"2019;" literal
text where a curly quote/apostrophe was meant - see
docs/superpowers/specs/2026-09-03-opportunities-synthesis-design.md's
sibling investigation for how this was found and verified against the real
corpus before writing any fix).

Backs up every row it's about to change to a JSON file first (id, source,
old_abstract) so the change is reversible, then applies the update.

Usage:
    uv run python scripts/backfill_abstract_entity_repair.py            # dry run
    uv run python scripts/backfill_abstract_entity_repair.py --apply    # writes
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from researchbridge.config import load_config
from researchbridge.connectors.base import clean_harvested_abstract
from researchbridge.db.models import Paper
from researchbridge.db.session import make_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        papers = session.query(Paper).filter(Paper.abstract.isnot(None)).all()
        changes = []
        for paper in papers:
            new_abstract = clean_harvested_abstract(paper.abstract)
            if new_abstract != paper.abstract:
                changes.append((paper, new_abstract))

        print(f"{len(changes)} / {len(papers)} papers would change")
        if not changes:
            return

        backup_path = Path(f"scripts/backfills/abstract_entity_repair_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                [
                    {"id": str(paper.id), "source": paper.source, "old_abstract": paper.abstract}
                    for paper, _new in changes
                ],
                indent=2,
            )
        )
        print(f"backed up {len(changes)} original abstracts to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for paper, new_abstract in changes:
            paper.abstract = new_abstract
        session.commit()
        print(f"updated {len(changes)} papers")


if __name__ == "__main__":
    main()
