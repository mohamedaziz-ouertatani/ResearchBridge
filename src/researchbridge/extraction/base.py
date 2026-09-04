"""Extractor interface: provider/model-agnostic claim extraction from a paper.

Any implementation - a local/free model, a mock/stub, or (optional, per the
blueprint's Free/Open-Source First constraint) a paid API - plugs in here
without the extraction pipeline needing to change. See ResearchBridge.md §29.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from researchbridge.db.models import Paper

# The Sec 25/28 field set, in reading order - reused wherever claims need a
# sensible display order (the API) rather than whatever order they happen
# to be stored/queried in.
CLAIM_TYPE_ORDER = (
    "problem",
    "research_question",
    "method",
    "dataset",
    "main_contribution",
    "results",
    "limitations",
    "research_gap",
    "applications",
)

# Sec 25/28 field -> the fulltext/parse.py section names most likely to
# actually state that field, in priority order (Sec 46's "full-text-aware
# extraction" design). A field's tuple is a priority list, not a pool:
# extraction/sections.py::sentences_for_field returns the FIRST listed
# section that's present in a paper's sections dict, never merging
# sentences across sections for one field. "problem" has no entry - its
# first-sentence-of-abstract fallback has no full-text equivalent worth
# inventing (see heuristic.py). "references"/"acknowledgments" are never a
# target - neither carries claim-bearing prose.
FIELD_SECTIONS: dict[str, tuple[str, ...]] = {
    "research_question": ("introduction",),
    "method": ("methods", "experiments"),
    "dataset": ("methods", "experiments"),
    "main_contribution": ("introduction", "conclusion"),
    "results": ("results",),
    "limitations": ("discussion", "limitations", "conclusion"),
    "research_gap": ("conclusion", "discussion", "limitations"),
    "applications": ("discussion", "conclusion", "introduction"),
}


@dataclass
class ClaimCandidate:
    claim_type: str
    claim_text: str
    evidence_quote: str
    confidence: str
    section: str = "abstract"
    """Which section evidence_quote was drawn from - "abstract" (the only
    value before Sec 46) or one of fulltext/parse.py's real section names.
    extraction/pipeline.py uses this to know what text to ground the quote
    against and what to persist as Evidence.section."""


class Extractor(Protocol):
    extraction_method: str
    model_version: str

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        """Return candidate claims for a paper.

        `sections` is this paper's paper_fulltext.sections (Sec 46), or {}
        when no full-text row exists yet - the common case until
        rb-fulltext-fetch processes more of the corpus. evidence_quote must
        be a verbatim substring of whatever section produced it
        (paper.abstract when candidate.section == "abstract", else
        sections[candidate.section]) - the pipeline verifies this before
        persisting anything and rejects candidates that fail.
        """
        ...
