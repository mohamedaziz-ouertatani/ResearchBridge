"""One-off backfill (part 2 of 2, see backfill_abstract_entity_repair.py):
the same stripped-HTML-entity corruption that was in papers.abstract was
also copied verbatim into evidence.text and extracted_claims.text at
extraction time, and into research_assessments' own persisted text fields
(research_gap_text, comparison_summary, risks_and_limitations) at
assessment-build time - those are formatted strings built once and stored,
not re-derived live, so fixing the source abstract alone does not fix an
already-built assessment report.

Backs up every row it changes to a JSON file first, same as part 1.

Usage:
    uv run python scripts/backfill_evidence_entity_repair.py            # dry run
    uv run python scripts/backfill_evidence_entity_repair.py --apply    # writes
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
from researchbridge.db.models import Evidence, ExtractedClaim, ResearchAssessment
from researchbridge.db.session import make_engine


def _clean(text: str | None) -> str | None:
    """Same repair as clean_harvested_abstract(), but never returns None
    for non-empty input: evidence.text/extracted_claims.text are NOT NULL
    columns, unlike papers.abstract (which this function was written for -
    a degenerate "abstract" that's just whitespace/an unresolvable entity
    like "&nbsp" with no trailing ";" correctly becomes None there). Found
    live: a real evidence.text value was exactly "&nbsp" (no semicolon),
    which html.unescape() still resolves to a bare non-breaking space that
    then strips to empty - collapsing an existing NOT NULL row to NULL
    would violate the schema and destroy content outside this fix's scope
    (a pre-existing degenerate-text data quality issue, not something this
    backfill is trying to address). Skip repairing text where doing so
    would leave nothing behind; the original stays as-is."""
    if text is None:
        return None
    cleaned = clean_harvested_abstract(text)
    return cleaned if cleaned is not None else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        evidence_changes = []
        for row in session.query(Evidence).filter(Evidence.text.isnot(None)).all():
            new_text = _clean(row.text)
            if new_text != row.text:
                evidence_changes.append((row, new_text))

        claim_changes = []
        for row in session.query(ExtractedClaim).filter(ExtractedClaim.text.isnot(None)).all():
            new_text = _clean(row.text)
            if new_text != row.text:
                claim_changes.append((row, new_text))

        assessment_field_changes = []
        for row in session.query(ResearchAssessment).all():
            for field in ("research_gap_text", "comparison_summary", "risks_and_limitations"):
                old_value = getattr(row, field)
                new_value = _clean(old_value)
                if new_value != old_value:
                    assessment_field_changes.append((row, field, old_value, new_value))

        print(f"evidence: {len(evidence_changes)} rows would change")
        print(f"extracted_claims: {len(claim_changes)} rows would change")
        print(f"research_assessments: {len(assessment_field_changes)} fields would change")

        total = len(evidence_changes) + len(claim_changes) + len(assessment_field_changes)
        if total == 0:
            return

        backup_path = Path(f"scripts/backfills/evidence_entity_repair_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                {
                    "evidence": [{"id": str(r.id), "old_text": r.text} for r, _new in evidence_changes],
                    "extracted_claims": [{"id": str(r.id), "old_text": r.text} for r, _new in claim_changes],
                    "research_assessments": [
                        {"id": str(r.id), "field": field, "old_value": old}
                        for r, field, old, _new in assessment_field_changes
                    ],
                },
                indent=2,
            )
        )
        print(f"backed up {total} original values to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for row, new_text in evidence_changes:
            row.text = new_text
        for row, new_text in claim_changes:
            row.text = new_text
        for row, field, _old, new_value in assessment_field_changes:
            setattr(row, field, new_value)
        session.commit()
        print(f"updated {total} values")


if __name__ == "__main__":
    main()
