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
    "applications",
)


@dataclass
class ClaimCandidate:
    claim_type: str
    claim_text: str
    evidence_quote: str
    confidence: str


class Extractor(Protocol):
    extraction_method: str
    model_version: str

    def extract(self, paper: Paper) -> list[ClaimCandidate]:
        """Return candidate claims for a paper.

        evidence_quote must be a text span that actually appears in the
        paper's content (currently: the abstract) — the pipeline verifies
        this before persisting anything, and rejects candidates that fail.
        """
        ...
