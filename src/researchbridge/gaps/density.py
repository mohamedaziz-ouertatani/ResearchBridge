"""Corpus-wide gap density by seed-paper domain category.

candidate_gaps has no domain/category column of its own - density is
computed by joining through seed_paper_id -> papers -> paper_categories.
A gap counts once per category its seed paper has (a multi-category paper's
gap appears in every one of those buckets), and once per category, not once
per contributing paper - only the seed paper's categories are used, since
that's the only 1:1 join available (a gap can have many contributing
papers, each with its own categories).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import CandidateGap, Paper, PaperCategory

UNCATEGORIZED = "uncategorized"


@dataclass
class GapDensityBucket:
    category: str
    gap_count: int


def compute_gap_density(session: Session) -> list[GapDensityBucket]:
    rows = session.execute(
        select(PaperCategory.category, CandidateGap.id)
        .select_from(CandidateGap)
        .join(Paper, Paper.id == CandidateGap.seed_paper_id)
        .join(PaperCategory, PaperCategory.paper_id == Paper.id, isouter=True)
        .where(CandidateGap.status != "rejected")
    ).all()

    gap_ids_by_category: dict[str, set] = {}
    for category, gap_id in rows:
        bucket = category or UNCATEGORIZED
        gap_ids_by_category.setdefault(bucket, set()).add(gap_id)

    return sorted(
        (GapDensityBucket(category=category, gap_count=len(gap_ids)) for category, gap_ids in gap_ids_by_category.items()),
        key=lambda b: (-b.gap_count, b.category),
    )
