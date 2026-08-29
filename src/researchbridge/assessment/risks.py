"""Evidence-grounded risks/limitations assessment (blueprint Sec 2A/49).

Same pattern as assess_applications: strictly extractive, never
synthesized. Surfaces each relevant retrieved paper's own explicit
"limitations" extraction claim (Sec 28/29) - never infers a new risk by
combining or clustering across papers (that cross-paper synthesis is what
gaps/cluster.py already does for research_gap specifically; conflating it
here would double up on the gap engine's own job).

Relevance-gated the same way as novelty/gap/applications: a retrieved
paper only contributes if its distance is within RELEVANCE_DISTANCE
(novelty.py's calibrated FAR_DISTANCE), so an off-topic paper's stated
limitations never leak into the report.

Unlike potential_applications (JSONB list), risks_and_limitations is a
plain TEXT column, so this aggregates into one grounded summary string -
one line per contributing paper, nearest-first - rather than a list of
records.

coverage_gaps (optional, from assessment/coverage.py) is a structurally
SEPARATE list of dimension labels with little/no supporting literature -
an evidence-coverage OBSERVATION, never a risk statement. A missing
dimension is never auto-promoted into `text`, which stays 100% sourced
from papers' own explicit limitations claims, exactly as before this
field existed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.coverage import DimensionCoverage
from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

PaperWithDistance = tuple[uuid.UUID, float]

_GAP_STATUSES = frozenset({"not_found", "weak_evidence"})


@dataclass
class RisksResult:
    text: str | None
    evidence_ids: list[uuid.UUID]
    coverage_gaps: list[str] = field(default_factory=list)


def assess_risks(
    session: Session,
    papers_by_distance: list[PaperWithDistance],
    dimension_coverages: list[DimensionCoverage] | None = None,
) -> RisksResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns).

    dimension_coverages (optional): from assessment/coverage.py. Used ONLY
    to populate coverage_gaps - a structurally separate list of dimension
    labels with little/no supporting literature. This is an evidence-
    coverage OBSERVATION, never a risk statement: a missing dimension is
    never auto-promoted into text, which stays 100% sourced from papers'
    own explicit limitations claims, exactly as before this parameter
    existed."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    coverage_gaps = [c.dimension for c in (dimension_coverages or []) if c.status in _GAP_STATUSES]

    if not relevant_paper_ids:
        return RisksResult(text=None, evidence_ids=[], coverage_gaps=coverage_gaps)

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "limitations",
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return RisksResult(text=None, evidence_ids=[], coverage_gaps=coverage_gaps)

    by_paper_id: dict[uuid.UUID, list[tuple[str, uuid.UUID, str]]] = {}
    for paper_id, text, evidence_id, title in rows:
        by_paper_id.setdefault(paper_id, []).append((text, evidence_id, title))

    lines: list[str] = []
    evidence_ids: list[uuid.UUID] = []
    for paper_id in relevant_paper_ids:
        for text, evidence_id, title in by_paper_id.get(paper_id, []):
            lines.append(f'- {title}: "{text}"')
            evidence_ids.append(evidence_id)

    return RisksResult(text="\n".join(lines), evidence_ids=evidence_ids, coverage_gaps=coverage_gaps)
