"""Sentence splitting shared by every extractive baseline.

Both the heuristic (cue-phrase) and semantic (embedding-similarity)
extractors work by picking one sentence from the abstract per field -
sharing the split keeps their candidate pools identical, so a difference
in evaluation results reflects the selection method, not a difference in
what counted as a sentence.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
