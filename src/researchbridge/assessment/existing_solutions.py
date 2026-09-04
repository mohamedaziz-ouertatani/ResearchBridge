"""Evidence-grounded "what already exists" comparison (blueprint Sec 2A/3).

Previously `comparison_summary` was a flat per-paper dump of every non-stub
extracted claim, unfiltered by claim_type - including research_gap and
applications claims, which already have their own dedicated report
sections (assess_research_gap, assess_applications) built from the same
underlying claims, so a paper's claim could show up twice under two
different headings, and results/method/problem text sat under no
organizing question at all - just "here is everything extracted about
this paper," a dump rather than a comparison.

Restructured to answer three of the fixed questions this section exists
to answer (blueprint Sec 3): what problems have already been addressed,
which approaches/methods already exist, and what limitations remain.
research_gap and applications are deliberately excluded here - they stay
sole-sourced from assess_research_gap/assess_applications, which apply
their own relevance gating and (research_gap) explicit-vs-inferred
labeling; repeating them here under a different heading would be exactly
the double-reporting problem this rewrite removes. "What is missing
relative to the input" is the research-gap section's job, not this one's
- see build.py for how the two are laid out together in the report.

Organized by question, not by paper, but never synthesizes a new
sentence out of the underlying claims: each line is one paper's own
extracted text, verbatim, with its evidence_id preserved - the same
"no Grounding Illusion" rule as everywhere else in this package.

Relevance-gated the same way as novelty/gap/applications/risks: a
retrieved paper only contributes if its distance is within
RELEVANCE_DISTANCE (novelty.py's calibrated FAR_DISTANCE). Previously
this was the one section with no gate at all - every one of top_k's
retrieved papers contributed its claims regardless of distance, so a
paper only loosely on-topic (e.g. 0.38 cosine distance, well past every
sibling section's 0.65 cutoff) could still flood "Problems already
addressed" with a generic, low-confidence claim (often just the
HeuristicExtractor's first-sentence-of-abstract fallback - see
extraction/heuristic.py) alongside genuinely relevant papers' specific
claims. Gating here brings this section in line with every other one;
it does not change what counts as a claim or how one is extracted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE

ClaimRecord = tuple[str, str, uuid.UUID]  # (claim_type, text, evidence_id)
PaperClaims = tuple[str, float, list[ClaimRecord]]  # (paper_title, distance, claims)

# claim_type -> (section heading, position). Order matters: it's the
# order sections are printed in, and the order the user's five questions
# were asked in (problems, then approaches, then limitations).
_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("problems_addressed", "Problems already addressed", ("problem",)),
    ("existing_approaches", "Existing approaches / methods used", ("method", "dataset")),
    ("limitations", "Limitations that remain", ("limitations",)),
)


@dataclass
class ExistingSolutionsResult:
    text: str | None
    evidence_ids: list[uuid.UUID]


_EMPTY_RESULT = ExistingSolutionsResult(text=None, evidence_ids=[])


def build_existing_solutions(papers_with_claims: list[PaperClaims]) -> ExistingSolutionsResult:
    """papers_with_claims: (paper_title, distance, claims) per retrieved
    paper with real (non-stub) claims, in the caller's own relevance
    order - reused as-is, not re-sorted, so ordering stays consistent
    with the rest of the report. A paper whose distance exceeds
    RELEVANCE_DISTANCE contributes nothing, same gate every sibling
    section already applies."""
    in_gate = [
        (paper_title, claims) for paper_title, distance, claims in papers_with_claims if distance <= RELEVANCE_DISTANCE
    ]

    evidence_ids: list[uuid.UUID] = []
    blocks: list[str] = []

    for _key, heading, claim_types in _SECTIONS:
        lines: list[str] = []
        for paper_title, claims in in_gate:
            for claim_type, text, evidence_id in claims:
                if claim_type in claim_types:
                    lines.append(f'- "{paper_title}": {text}')
                    evidence_ids.append(evidence_id)
        if lines:
            blocks.append(heading + "\n" + "\n".join(lines))

    if not blocks:
        return _EMPTY_RESULT
    return ExistingSolutionsResult(text="\n\n".join(blocks), evidence_ids=evidence_ids)
