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
