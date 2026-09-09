from __future__ import annotations

import pytest
import requests

from researchbridge.assessment.dimensions import IdeaDimension
from researchbridge.assessment.dimensions_llm import (
    DimensionExtractionUnavailable,
    extract_dimensions_via_llm,
    extract_dimensions_with_fallback,
    parse_dimensions_response,
)


def test_parses_one_dimension_per_line() -> None:
    response = "1. privacy-preserving federated learning\n2. financial fraud detection\n3. concept drift"
    dimensions = parse_dimensions_response(response, max_dimensions=8)
    assert dimensions == [
        IdeaDimension(label="privacy-preserving federated learning"),
        IdeaDimension(label="financial fraud detection"),
        IdeaDimension(label="concept drift"),
    ]


def test_parse_raises_on_empty_response() -> None:
    with pytest.raises(ValueError):
        parse_dimensions_response("", max_dimensions=8)


def test_extract_dimensions_via_llm_raises_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    with pytest.raises(DimensionExtractionUnavailable):
        extract_dimensions_via_llm("a privacy-preserving federated learning system")


def test_fallback_uses_rake_when_llm_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    dimensions = extract_dimensions_with_fallback("a privacy-preserving federated learning system for fraud")
    assert len(dimensions) > 0  # RAKE's existing deterministic output, never empty for real prose


def test_fallback_uses_rake_when_llm_unreachable(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")

    def _raise(*args, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("researchbridge.assessment.dimensions_llm.requests.post", _raise)
    dimensions = extract_dimensions_with_fallback("a privacy-preserving federated learning system for fraud")
    assert len(dimensions) > 0


def test_duplicate_dimension_labels_are_collapsed() -> None:
    """Repeated labels inflate the denominator novelty.py divides by, so
    the same concept counted three times makes an idea look less covered
    than it is. Found live 2026-09-09: a prompt-injection input drove the
    model to repeat "Graph neural networks" three times, yielding n=8
    over 3 distinct concepts."""
    response = "\n".join(
        [
            "1. Graph neural networks",
            "2. Molecular property prediction",
            "3. Contrastive objective",
            "4. Graph neural networks",
            "5. Molecular property prediction",
            "6. Contrastive objective",
        ]
    )

    dimensions = parse_dimensions_response(response, max_dimensions=8)

    assert [d.label for d in dimensions] == [
        "Graph neural networks",
        "Molecular property prediction",
        "Contrastive objective",
    ]


def test_duplicate_detection_ignores_case_and_surrounding_whitespace() -> None:
    response = "1. Protein Contact Maps\n2. protein contact maps  \n3. Contrastive objective"

    dimensions = parse_dimensions_response(response, max_dimensions=8)

    assert [d.label for d in dimensions] == ["Protein Contact Maps", "Contrastive objective"]


def test_generic_filler_labels_are_dropped() -> None:
    """The prompt already forbids generic verbs and filler, and the model
    still emits them. Every filler label occupies a slot and scores
    not_found, biasing novelty upward. Found live 2026-09-09: a clean
    drug-target idea yielded "Encoding" and "Jointly" as 2 of its 8
    dimensions."""
    response = "\n".join(
        [
            "1. Graph neural network",
            "2. Encoding",
            "3. Jointly",
            "4. Evaluation",
            "5. Training",
            "6. ChEMBL bioactivity data",
        ]
    )

    dimensions = parse_dimensions_response(response, max_dimensions=8)

    assert [d.label for d in dimensions] == ["Graph neural network", "ChEMBL bioactivity data"]


def test_a_multi_word_phrase_containing_a_filler_word_is_kept() -> None:
    response = "1. Adversarial training\n2. Model evaluation protocol"

    dimensions = parse_dimensions_response(response, max_dimensions=8)

    assert [d.label for d in dimensions] == ["Adversarial training", "Model evaluation protocol"]


def test_blank_idea_text_never_reaches_the_model() -> None:
    """With no idea to extract from, the model answers by summarising its
    own instructions, and those get persisted as the user's research
    dimensions. Found live 2026-09-09: whitespace-only input produced
    dimensions "User-submitted content", "Nouns/noun phrases", "Grounded
    in wording" and "Avoid invented terminology" - the system prompt read
    back verbatim into a user-facing report."""
    with pytest.raises(DimensionExtractionUnavailable):
        extract_dimensions_via_llm("   \n\t  ")


def test_blank_idea_text_falls_back_without_calling_the_model() -> None:
    assert extract_dimensions_with_fallback("   ") == []
