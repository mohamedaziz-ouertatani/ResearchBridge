"""Uploaded-document research representation (blueprint Sec 2A).

An uploaded document's raw text isn't a good retrieval query directly -
it's a whole paper dump, not a focused query the way an idea's raw_text
already is. Before search_by_text runs, build a compact "research
representation" from the upload instead, reusing the same HybridExtractor
already used for corpus papers (Sec 28/29) rather than any new extraction
logic.

This never persists anything: no Evidence/ExtractedClaim rows get created
for the input (ResearchInput has no such linkage in the schema, and none
is needed). A transient, un-saved Paper-shaped object is passed through
the extractor purely to get candidate claims to build a query string
from - nothing here touches the database.

Only problem/research_question/method/main_contribution feed the query -
the fields that describe what the work IS, not incidental details like
dataset/limitations/results that would just add retrieval noise.

Falls back to the raw text itself if extraction yields nothing usable
(e.g. the abstract has no sentence-level structure the heuristic
extractor can work with) - never an empty query.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.hybrid import HybridExtractor

_REPRESENTATION_CLAIM_TYPES = ("problem", "research_question", "method", "main_contribution")


def build_research_representation(raw_text: str, embedder: Embedder) -> str:
    if not raw_text or not raw_text.strip():
        return raw_text

    transient_paper = Paper(abstract=raw_text)
    candidates = HybridExtractor(embedder).extract(transient_paper, {})

    # A short or single-sentence input can have the same sentence chosen for
    # more than one field (e.g. the heuristic's "problem" fallback and the
    # semantic extractor's nearest-sentence pick can coincide) - dedupe
    # while preserving field order, so the query isn't padded with repeats.
    seen: set[str] = set()
    parts: list[str] = []
    for candidate in candidates:
        if candidate.claim_type not in _REPRESENTATION_CLAIM_TYPES or candidate.claim_text in seen:
            continue
        seen.add(candidate.claim_text)
        parts.append(candidate.claim_text)

    return "\n".join(parts) if parts else raw_text
