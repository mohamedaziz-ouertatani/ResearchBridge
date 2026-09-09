"""Report how much of the corpus carries each extraction claim type.

Motivation (2026-09-09): live testing found `potential_applications` empty
on almost every assessment, and `potential_opportunities` NULL on all 246
assessments ever built. Neither was a scoring bug. Both trace to a single
number that nothing surfaced: an "applications" claim exists on roughly 2%
of papers, and opportunity synthesis only runs when applications is
"found", so the whole downstream stage is starved by extraction coverage.

That number was invisible until someone queried for it by hand. This
script makes it a command, so the decision to re-extract (or to loosen
extraction/validation.py's gates) is made against real coverage rather
than a guess.

Read-only: issues SELECTs and prints. Never writes.

    python scripts/report_claim_type_coverage.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import text

from researchbridge.config import load_config
from researchbridge.db.session import make_engine

# Claim types whose absence silently disables a downstream assessment
# stage, rather than merely leaving one report section empty.
_GATING_CLAIM_TYPES = {
    "applications": "gates potential_applications AND opportunity synthesis (assessment/build.py)",
    "research_gap": "gates the explicit-gap branch of assessment/gap.py",
    "limitations": "the sole source for risks_and_limitations",
}


def main() -> int:
    load_config()
    engine = make_engine()
    with engine.connect() as connection:
        total_papers = connection.execute(
            text("select count(*) from papers where excluded_at is null")
        ).scalar_one()
        rows = connection.execute(
            text(
                """
                select ec.claim_type,
                       count(*) as claims,
                       count(distinct ec.paper_id) as papers
                from extracted_claims ec
                join evidence ev on ev.id = ec.evidence_id
                join papers p on p.id = ec.paper_id
                where ev.extraction_method != 'stub' and p.excluded_at is null
                group by ec.claim_type
                order by papers desc
                """
            )
        ).all()

    if not total_papers:
        print("no papers in the corpus - nothing to report")
        return 1

    print(f"corpus: {total_papers:,} papers (excluding withdrawn)\n")
    print(f"{'claim type':<20} {'papers':>9} {'coverage':>9} {'claims':>10}")
    print("-" * 52)
    for claim_type, claims, papers in rows:
        print(f"{claim_type:<20} {papers:>9,} {papers / total_papers:>8.1%} {claims:>10,}")

    print()
    covered = {claim_type: papers for claim_type, _claims, papers in rows}
    for claim_type, consequence in _GATING_CLAIM_TYPES.items():
        papers = covered.get(claim_type, 0)
        print(f"{claim_type}: {papers / total_papers:.1%} of papers - {consequence}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
