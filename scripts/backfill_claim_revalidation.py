"""One-off targeted re-validation backfill (as an alternative to a full
wipe-and-re-ingest): re-runs extraction/validation.py::validate_claim_type
against every existing research_gap and applications claim already in the
database, using TODAY's validation rules - not the rules that were live
when each claim was originally extracted.

Why these two claim types specifically: both have had multiple real,
verified false-positive fixes land recently (the "gap between"/"computes
the gap between"/"open question answering" research_gap fixes; the
enumeration/"for a given user"/deployment-by-authority applications
fixes), and both persist a validation_tier that can now be checked against
current code. The other four validatable claim types (problem, method,
results, limitations) never persist a tier at all (see ExtractedClaim.
validation_tier's docstring) and have no similarly-verified recent fix to
re-check against, so re-validating them here would be guessing rather
than targeted.

Nothing here re-fetches from any external API or re-runs embeddings -
this only re-checks text already in the database against the validator
already in this codebase. A claim whose text no longer passes validation
is deleted (matching what the live extraction pipeline already does for a
freshly-extracted invalid claim: reject, never persist) - the linked
Evidence row is left alone, since the quoted text itself isn't fabricated,
only its claim-type label was wrong, and other real assessments may not
reference it at all. A claim that still passes but whose stored tier is
stale gets its validation_tier corrected in place.

Deliberately out of scope: already-built ResearchAssessment rows whose
persisted text fields (research_gap_text, comparison_summary, ...) were
computed from a claim this backfill removes are NOT retroactively
rewritten - that requires re-running the relevant assess_* logic per
assessment, a materially bigger operation than re-validating claims.

Usage:
    uv run python scripts/backfill_claim_revalidation.py            # dry run
    uv run python scripts/backfill_claim_revalidation.py --apply    # writes
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from researchbridge.config import load_config
from researchbridge.db.models import ExtractedClaim
from researchbridge.db.session import make_engine
from researchbridge.extraction.validation import validate_claim_type

CLAIM_TYPES = ("research_gap", "applications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        claims = session.query(ExtractedClaim).filter(ExtractedClaim.claim_type.in_(CLAIM_TYPES)).all()

        to_delete: list[tuple[ExtractedClaim, str]] = []
        to_retier: list[tuple[ExtractedClaim, str | None, str | None]] = []

        for claim in claims:
            result = validate_claim_type(claim.claim_type, claim.text)
            if not result.is_valid:
                to_delete.append((claim, result.reason or "no reason given"))
            elif result.tier != claim.validation_tier:
                to_retier.append((claim, claim.validation_tier, result.tier))

        print(f"checked {len(claims)} claims ({', '.join(CLAIM_TYPES)})")
        print(f"{len(to_delete)} now fail validation and would be deleted")
        print(f"{len(to_retier)} pass but have a stale/missing tier and would be corrected")

        by_type_deleted: dict[str, int] = {}
        for claim, _reason in to_delete:
            by_type_deleted[claim.claim_type] = by_type_deleted.get(claim.claim_type, 0) + 1
        for claim_type, count in by_type_deleted.items():
            print(f"  delete: {claim_type}: {count}")

        if not to_delete and not to_retier:
            return

        backup_path = Path(f"scripts/backfills/claim_revalidation_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                {
                    "deleted": [
                        {
                            "id": str(claim.id),
                            "paper_id": str(claim.paper_id),
                            "claim_type": claim.claim_type,
                            "text": claim.text,
                            "evidence_id": str(claim.evidence_id),
                            "old_validation_tier": claim.validation_tier,
                            "rejection_reason": reason,
                        }
                        for claim, reason in to_delete
                    ],
                    "retiered": [
                        {"id": str(claim.id), "old_tier": old, "new_tier": new}
                        for claim, old, new in to_retier
                    ],
                },
                indent=2,
            )
        )
        print(f"backed up {len(to_delete) + len(to_retier)} affected rows to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for claim, _old, new in to_retier:
            claim.validation_tier = new
        for claim, _reason in to_delete:
            session.delete(claim)
        session.commit()
        print(f"deleted {len(to_delete)} claims, corrected {len(to_retier)} tiers")


if __name__ == "__main__":
    main()
