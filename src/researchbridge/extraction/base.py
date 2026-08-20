"""Extractor interface: provider/model-agnostic claim extraction from a paper.

Any implementation - a local/free model, a mock/stub, or (optional, per the
blueprint's Free/Open-Source First constraint) a paid API - plugs in here
without the extraction pipeline needing to change. See ResearchBridge.md §29.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from researchbridge.db.models import Paper


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
