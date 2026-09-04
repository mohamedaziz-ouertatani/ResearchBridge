"""Section-aware sentence lookup shared by HeuristicExtractor and
SemanticExtractor (Sec 46 full-text-aware extraction).
"""

from __future__ import annotations

from researchbridge.extraction.base import FIELD_SECTIONS
from researchbridge.extraction.sentences import split_sentences


def sentences_for_field(field: str, sections: dict[str, str]) -> list[tuple[str, str]]:
    """Returns [(sentence, section_name), ...] for the first of `field`'s
    FIELD_SECTIONS entries that's present (and non-blank) in `sections` -
    section priority, not a merged pool: sentences are drawn from exactly
    one section per call, never combined across the tuple. Empty list if
    `field` has no FIELD_SECTIONS entry, or none of its target sections
    are present - the caller's cue to fall back to the abstract.
    """
    for section_name in FIELD_SECTIONS.get(field, ()):
        text = sections.get(section_name)
        if text and text.strip():
            return [(sentence, section_name) for sentence in split_sentences(text)]
    return []
