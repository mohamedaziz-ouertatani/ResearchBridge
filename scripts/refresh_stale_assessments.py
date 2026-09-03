"""Item 5 of the assessment hardening list: refresh every ResearchAssessment
built before today's validator fixes (Gate 1 weak/strong tagging, the QPE
root-cause fix, Gate 2's weak-tier overlap threshold, dimensions.py's
punctuation-boundary fix) so the report a user sees reflects current code,
not whatever was live when it was first assessed.

Not a claim-level backfill like backfill_claim_revalidation.py - a
ResearchAssessment is a computed snapshot (comparison_summary, novelty_level,
research_gap_text, potential_applications, ...), not raw data, so "fixing"
it means recomputing it, not patching a field in place.

Reuses the exact mechanism the app's own "Rerun" button already calls
(assessment_routes.py::rerun_assessment -> assessment/build.py::
build_assessment): always creates a NEW ResearchAssessment row for the same
research_input rather than overwriting the old one - the list page already
collapses to "latest per research_input" (see list_assessments), so a fresh
rerun naturally supersedes the stale row without deleting anything, keeping
assessment history honest exactly the way a manual Rerun click would.
Verified before running: none of the 32 existing rows have human_reviewed=
True, so no review state is at risk of being superseded by an unreviewed
fresh row. No LLM cost either - opportunities synthesis is a separate,
on-demand endpoint build_assessment never calls itself.

Usage:
    uv run python scripts/refresh_stale_assessments.py
"""

from __future__ import annotations

import os
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.build import build_assessment
from researchbridge.config import load_config
from researchbridge.db.models import ResearchAssessment, ResearchInput
from researchbridge.db.session import make_engine
from researchbridge.embedding.model import SentenceTransformerEmbedder


def _summary(a: ResearchAssessment) -> dict:
    return {
        "novelty": a.novelty_level,
        "gap_source": a.research_gap_source,
        "applications": len(a.potential_applications) if a.potential_applications else 0,
        "feasibility": a.technical_feasibility_level,
        "recommendation": a.recommendation,
        "confidence": a.confidence,
    }


def main() -> None:
    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])
    embedder = SentenceTransformerEmbedder()

    with Session(engine) as session:
        latest_by_input: dict[uuid.UUID, ResearchAssessment] = {}
        for a in session.execute(
            select(ResearchAssessment).order_by(ResearchAssessment.completed_at)
        ).scalars():
            latest_by_input[a.research_input_id] = a

        print(f"refreshing {len(latest_by_input)} research inputs' assessments\n")

        changed = 0
        for research_input_id, before in latest_by_input.items():
            research_input = session.get(ResearchInput, research_input_id)
            preview = research_input.raw_text[:70].replace("\n", " ")
            before_summary = _summary(before)

            after = build_assessment(session, research_input_id, embedder)
            after_summary = _summary(after)

            print(f"=== {research_input_id}  {preview!r}")
            if before_summary == after_summary:
                print("    unchanged")
            else:
                changed += 1
                for key in before_summary:
                    if before_summary[key] != after_summary[key]:
                        print(f"    {key}: {before_summary[key]!r} -> {after_summary[key]!r}")

        print(f"\n{changed}/{len(latest_by_input)} research inputs got a materially different assessment")


if __name__ == "__main__":
    main()
