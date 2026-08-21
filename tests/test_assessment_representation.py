from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from researchbridge.assessment.representation import build_research_representation
from researchbridge.db.models import EMBEDDING_DIM


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


_REALISTIC_DOCUMENT_TEXT = (
    "Detecting fraud in financial transactions remains a major challenge for banks. "
    "We propose a graph transformer architecture for real-time fraud inference. "
    "Our main contribution is a new attention mechanism for temporal transaction graphs. "
    "However, our evaluation is limited to offline historical datasets."
)


def test_extracts_a_representation_from_realistic_document_text() -> None:
    representation = build_research_representation(_REALISTIC_DOCUMENT_TEXT, FakeEmbedder())

    assert "graph transformer architecture" in representation
    assert "new attention mechanism" in representation


def test_representation_excludes_limitations_and_dataset_fields() -> None:
    representation = build_research_representation(_REALISTIC_DOCUMENT_TEXT, FakeEmbedder())

    # limitations aren't part of what makes a good retrieval query
    assert "offline historical datasets" not in representation


def test_dedupes_when_every_field_lands_on_the_same_lone_sentence() -> None:
    # a single-sentence input has no cue-phrase match, so heuristic's
    # "problem" fallback and semantic's nearest-sentence pick can both
    # choose it for multiple fields - deduped, it should just be that
    # one sentence, not the same line repeated per field
    text = "asdf qwerty zzyzx plugh xyzzy."

    representation = build_research_representation(text, FakeEmbedder())

    assert representation == text


def test_falls_back_to_raw_text_for_empty_string() -> None:
    representation = build_research_representation("", FakeEmbedder())

    assert representation == ""


def test_falls_back_to_raw_text_for_whitespace_only() -> None:
    representation = build_research_representation("   \n  ", FakeEmbedder())

    assert representation == "   \n  "
