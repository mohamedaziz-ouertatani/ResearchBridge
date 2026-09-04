"""One-off targeted re-validation backfill for "dataset" claims, mirroring
scripts/backfill_claim_revalidation.py's own pattern: re-runs extraction/
validation.py::validate_claim_type against every existing "dataset" claim
in the database, using TODAY's validation rules.

Why this claim type, run separately from the existing script: "dataset"
was not in VALIDATABLE_CLAIM_TYPES at all until 2026-09-04 (see
validation.py's own module docstring) - every "dataset" claim ever
extracted was accepted unconditionally, so unlike research_gap/
applications (which have been validated from day one and only need
re-checking against fixes to an existing gate), this is the FIRST
validation pass these claims have ever seen. Confirmed live: a generic
survey sentence with zero dataset content ("By harnessing machine
learning algorithms... AI enables the analysis of complex medical data")
was extracted as "dataset" and, via assessment/feasibility.py, alone
drove a false "technical_feasibility_level: high" verdict for an
unrelated, vague idea.

"dataset" never persists a validation_tier (see ExtractedClaim.
validation_tier's docstring - only research_gap/applications do), so this
only ever deletes claims that now fail validation; there is no re-tier
path to mirror.

Nothing here re-fetches from any external API or re-runs embeddings -
this only re-checks text already in the database against the validator
already in this codebase. A claim whose text no longer passes validation
is deleted (matching what the live extraction pipeline already does for a
freshly-extracted invalid claim: reject, never persist) - the linked
Evidence row is left alone, since the quoted text itself isn't
fabricated, only its claim-type label was wrong.

Deliberately out of scope: already-built ResearchAssessment rows whose
persisted fields (technical_feasibility_level/reasoning, potential_
applications, ...) were computed using a claim this backfill removes are
NOT retroactively rewritten - that requires re-running the relevant
assess_* logic per assessment (the API's own POST /{id}/rerun endpoint
does this on demand), a materially bigger operation than re-validating
claims.

Usage:
    uv run python scripts/backfill_dataset_claim_revalidation.py            # dry run
    uv run python scripts/backfill_dataset_claim_revalidation.py --apply    # writes
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

CLAIM_TYPES = ("dataset",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        claims = session.query(ExtractedClaim).filter(ExtractedClaim.claim_type.in_(CLAIM_TYPES)).all()

        to_delete: list[tuple[ExtractedClaim, str]] = []

        for claim in claims:
            result = validate_claim_type(claim.claim_type, claim.text)
            if not result.is_valid:
                to_delete.append((claim, result.reason or "no reason given"))

        print(f"checked {len(claims)} claims ({', '.join(CLAIM_TYPES)})")
        print(f"{len(to_delete)} now fail validation and would be deleted")

        if not to_delete:
            return

        backup_path = Path(f"scripts/backfills/dataset_claim_revalidation_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
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
                            "rejection_reason": reason,
                        }
                        for claim, reason in to_delete
                    ]
                },
                indent=2,
            )
        )
        print(f"backed up {len(to_delete)} affected rows to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for claim, _reason in to_delete:
            session.delete(claim)
        session.commit()
        print(f"deleted {len(to_delete)} claims")


if __name__ == "__main__":
    main()
