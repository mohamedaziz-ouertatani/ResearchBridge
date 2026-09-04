"""A deliberately trivial Extractor used to exercise the pipeline mechanics.

This does NOT perform real extraction methodology - no NLP, no LLM, no
keyword heuristics. It exists to prove the evidence-grounding pipeline
(verify -> persist -> track run/errors) works end-to-end, not to produce
usable research content.

Every row it produces carries extraction_method="stub" (see Evidence's
docstring in db/models.py) so it can never be mistaken for real analysis:
never use stub output as benchmark ground truth, and always filter it out
of any real evaluation or downstream research-intelligence query.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.extraction.base import ClaimCandidate

STUB_MODEL_VERSION = "stub-v1"


class StubExtractor:
    extraction_method = "stub"
    model_version = STUB_MODEL_VERSION

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        if not paper.abstract or not paper.abstract.strip():
            return []

        abstract = paper.abstract.strip()
        return [
            ClaimCandidate(
                claim_type="contribution",
                claim_text=abstract,
                evidence_quote=abstract,
                confidence="medium",  # "extracted from an abstract only" - see ResearchBridge.md §15
            )
        ]
