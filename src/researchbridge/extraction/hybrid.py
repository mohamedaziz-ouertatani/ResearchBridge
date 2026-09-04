"""Combines the heuristic and semantic extractors (Sec 26 Improvement step).

Per-field comparison against the benchmark showed the two baselines win on
almost entirely different fields: heuristic on problem/method/limitations,
semantic on research_question/main_contribution/results/applications (see
the commit that added semantic.py). It is tempting to just hardcode that
per-field routing - but that would mean choosing rules by reading the
score table off the exact 40 papers this extractor gets re-evaluated
against, which quietly launders an inflated number rather than measuring
anything. The routing below is deliberately NOT built that way.

Two rules, both justified without looking at the benchmark's ground truth:

1. A literal cue-phrase hit (heuristic confidence in "medium"/"high" - see
   below) is trusted over semantic similarity when one exists - an
   explicit, specific textual signal is stronger evidence than a general
   similarity score when both are available.

2. "problem" specifically always prefers the heuristic's answer, including
   its low-confidence first-sentence fallback. This is because abstracts
   conventionally open with the problem/motivation before anything else -
   a documented property of how scientific abstracts are written, not
   something read off this benchmark's answers.

Everywhere else, semantic fills in where heuristic found no cue-phrase
match at all - the situation semantic exists for (Sec 26): fields whose
real-world phrasing varies too much for a fixed phrase list.

Rule 1's confidence check covers both "medium" (an abstract-only cue-phrase
hit) and "high" (a full-text, named-section cue-phrase hit - Sec 46) -
both mean the same thing this rule cares about, "an explicit cue-phrase
match exists", so a full-text heuristic hit is preferred over semantic
exactly like an abstract one always was.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.heuristic import HeuristicExtractor
from researchbridge.extraction.semantic import SemanticExtractor

HYBRID_MODEL_VERSION = "hybrid-v1"


class HybridExtractor:
    extraction_method = "hybrid"
    model_version = HYBRID_MODEL_VERSION

    def __init__(self, embedder: Embedder) -> None:
        self._heuristic = HeuristicExtractor()
        self._semantic = SemanticExtractor(embedder)

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        heuristic_by_field = {c.claim_type: c for c in self._heuristic.extract(paper, sections)}
        semantic_by_field = {c.claim_type: c for c in self._semantic.extract(paper, sections)}

        fields = heuristic_by_field.keys() | semantic_by_field.keys()
        chosen = (
            self._choose(field, heuristic_by_field.get(field), semantic_by_field.get(field))
            for field in fields
        )
        return [c for c in chosen if c is not None]

    def _choose(
        self, field: str, heuristic: ClaimCandidate | None, semantic: ClaimCandidate | None
    ) -> ClaimCandidate | None:
        if field == "problem":
            return heuristic or semantic
        if heuristic is not None and heuristic.confidence in ("medium", "high"):
            return heuristic
        return semantic or heuristic
