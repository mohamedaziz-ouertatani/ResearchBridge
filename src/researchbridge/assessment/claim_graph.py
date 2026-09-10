"""Per-assessment claim/evidence/gap graph (Interactive Claim & Evidence Graph spec).

Computed on demand, same "recompute rather than persist a derived view"
pattern as assessment/graph.py. ADDRESSES_GAP edges are synthesized, not
read from a column - within one assessment's scope, every assessment-derived
claim is by definition reasoning about that assessment's own
candidate_gap_id, so one such edge is added per assessment claim -> gap. The
gap's own describing claim (source_table="candidate_gaps", see
gaps/claims.py) is included as an ordinary claim node with its own evidence
via claim_evidence - CandidateGapEvidence is redundant for this graph
because gaps/claims.py::save_claim_for_gap already mirrors every
CandidateGapEvidence link into a claim_evidence "supports" row when it
creates that claim.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    ClaimEvidence,
    Evidence,
    Paper,
    PaperCategory,
    ResearchAssessment,
)


@dataclass
class ClaimGraphNode:
    id: str
    kind: Literal["claim", "evidence", "gap"]
    label: str
    claim_type: str | None = None
    confidence: str | None = None
    status: str | None = None
    paper_id: str | None = None
    paper_title: str | None = None
    section: str | None = None
    gap_status: str | None = None
    categories: list[str] = field(default_factory=list)


@dataclass
class ClaimGraphEdge:
    source: str
    target: str
    relationship: Literal["supports", "contradicts", "contextualizes", "addresses_gap"]


@dataclass
class ClaimEvidenceGraph:
    nodes: list[ClaimGraphNode]
    edges: list[ClaimGraphEdge]


def build_claim_evidence_graph(session: Session, assessment: ResearchAssessment) -> ClaimEvidenceGraph:
    assessment_claims = list(
        session.execute(
            select(AnalysisClaim).where(
                AnalysisClaim.source_table == "research_assessments",
                AnalysisClaim.source_id == assessment.id,
            )
        ).scalars()
    )
    assessment_claim_ids = {c.id for c in assessment_claims}

    gap: CandidateGap | None = None
    gap_claim: AnalysisClaim | None = None
    if assessment.candidate_gap_id is not None:
        gap = session.get(CandidateGap, assessment.candidate_gap_id)
        if gap is not None:
            gap_claim = session.execute(
                select(AnalysisClaim).where(
                    AnalysisClaim.source_table == "candidate_gaps", AnalysisClaim.source_id == gap.id
                )
            ).scalar_one_or_none()

    all_claims = list(assessment_claims)
    if gap_claim is not None:
        all_claims.append(gap_claim)

    claim_ids = [c.id for c in all_claims]
    evidence_rows = (
        session.execute(
            select(ClaimEvidence, Evidence, Paper.title)
            .join(Evidence, Evidence.id == ClaimEvidence.evidence_id)
            .join(Paper, Paper.id == Evidence.paper_id)
            .where(ClaimEvidence.claim_id.in_(claim_ids))
        ).all()
        if claim_ids
        else []
    )

    evidence_nodes: dict[uuid.UUID, ClaimGraphNode] = {}
    edges: list[ClaimGraphEdge] = []
    for link, evidence, paper_title in evidence_rows:
        if evidence.id not in evidence_nodes:
            evidence_nodes[evidence.id] = ClaimGraphNode(
                id=str(evidence.id), kind="evidence", label=evidence.text,
                paper_id=str(evidence.paper_id), paper_title=paper_title, section=evidence.section,
            )
        edges.append(
            ClaimGraphEdge(source=str(link.claim_id), target=str(evidence.id), relationship=link.relationship)
        )

    claim_nodes = [
        ClaimGraphNode(
            id=str(c.id), kind="claim", label=c.claim_text,
            claim_type=c.claim_type, confidence=c.confidence, status=c.status,
        )
        for c in all_claims
    ]

    nodes: list[ClaimGraphNode] = [*claim_nodes, *evidence_nodes.values()]

    if gap is not None:
        categories = [
            row
            for row in session.execute(
                select(PaperCategory.category).where(PaperCategory.paper_id == gap.seed_paper_id)
            ).scalars()
        ]
        nodes.append(
            ClaimGraphNode(
                id=str(gap.id), kind="gap", label=gap.observation, gap_status=gap.gap_status, categories=categories,
            )
        )
        for claim_id in assessment_claim_ids:
            edges.append(ClaimGraphEdge(source=str(claim_id), target=str(gap.id), relationship="addresses_gap"))

    return ClaimEvidenceGraph(nodes=nodes, edges=edges)
