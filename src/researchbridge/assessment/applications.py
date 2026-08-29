"""Evidence-grounded application discovery (blueprint Sec 2A/33).

Deliberately narrow, like assess_research_gap: only surfaces applications
explicitly stated by a relevant retrieved paper's own "applications"
extraction claim (Sec 28/29's ninth extraction field) - never synthesizes
a novel application by combining or clustering across papers. That
further interpretive leap is "Potential Product/Technology Opportunities"
(Sec 49) - a separate, not-yet-built report field. Conflating the two
would blur Sec 34's fact-vs-opportunity distinction: an application an
author already states is a fact about that paper, not an inference.

Relevance-gated the same way as assess_research_gap: a retrieved paper
only contributes if its distance is within RELEVANCE_DISTANCE (novelty
.py's calibrated FAR_DISTANCE), so an off-topic paper's applications never
leak into the report.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

PaperWithDistance = tuple[uuid.UUID, float]


@dataclass
class ApplicationRecord:
    application: str
    source_paper: str
    paper_id: uuid.UUID
    evidence_id: uuid.UUID


@dataclass
class ApplicationsResult:
    applications: list[ApplicationRecord] | None
    evidence_ids: list[uuid.UUID]
    status: str = "not_assessed"  # "not_assessed" | "no_evidence" | "found"


_NOT_ASSESSED_RESULT = ApplicationsResult(applications=None, evidence_ids=[], status="not_assessed")
_NO_EVIDENCE_RESULT = ApplicationsResult(applications=[], evidence_ids=[], status="no_evidence")


def assess_applications(session: Session, papers_by_distance: list[PaperWithDistance]) -> ApplicationsResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns)."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    if not relevant_paper_ids:
        return _NOT_ASSESSED_RESULT

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "applications",
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return _NO_EVIDENCE_RESULT

    by_paper_id: dict[uuid.UUID, list[tuple[str, uuid.UUID, str]]] = {}
    for paper_id, text, evidence_id, title in rows:
        by_paper_id.setdefault(paper_id, []).append((text, evidence_id, title))

    applications: list[ApplicationRecord] = []
    evidence_ids: list[uuid.UUID] = []
    for paper_id in relevant_paper_ids:
        for text, evidence_id, title in by_paper_id.get(paper_id, []):
            applications.append(
                ApplicationRecord(application=text, source_paper=title, paper_id=paper_id, evidence_id=evidence_id)
            )
            evidence_ids.append(evidence_id)

    return ApplicationsResult(applications=applications, evidence_ids=evidence_ids, status="found")
