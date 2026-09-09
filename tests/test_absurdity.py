from __future__ import annotations

from unittest.mock import Mock

import pytest

from researchbridge.assessment.absurdity import (
    AbsurdityCheckUnavailable,
    check_for_irreconcilable_concepts,
    parse_response,
)

# --- parse_response -------------------------------------------------------


def test_parse_response_plausible_is_not_flagged() -> None:
    result = parse_response("PLAUSIBLE")

    assert result.flagged is False
    assert result.reason is None


def test_parse_response_is_case_insensitive() -> None:
    result = parse_response("plausible")

    assert result.flagged is False


def test_parse_response_requires_review_is_flagged_with_reason() -> None:
    result = parse_response(
        "REQUIRES_REVIEW: mixes non-Euclidean graph geometry with zero-knowledge proofs with no precedent"
    )

    assert result.flagged is True
    assert "zero-knowledge" in result.reason


def test_parse_response_strips_surrounding_whitespace() -> None:
    result = parse_response("  PLAUSIBLE  \n")

    assert result.flagged is False


def test_parse_response_collapses_internal_whitespace_in_reason() -> None:
    result = parse_response("REQUIRES_REVIEW: two   domains\nthat do not combine")

    assert result.reason == "two domains that do not combine"


def test_parse_response_raises_on_empty_reason() -> None:
    with pytest.raises(ValueError, match="no reason"):
        parse_response("REQUIRES_REVIEW:")


def test_parse_response_raises_on_unrecognized_text() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        parse_response("I think this is fine I guess")


# --- check_for_irreconcilable_concepts (Ollama call mocked) --------------


def _mock_ollama_response(monkeypatch: pytest.MonkeyPatch, content: str) -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": content}}
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("researchbridge.assessment.absurdity.requests.post", mock_post)
    return mock_post


def test_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")

    with pytest.raises(AbsurdityCheckUnavailable, match="not enabled"):
        check_for_irreconcilable_concepts("some idea")


def test_raises_when_idea_text_is_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")

    with pytest.raises(AbsurdityCheckUnavailable, match="no idea text"):
        check_for_irreconcilable_concepts("   ")


def test_returns_not_flagged_for_a_plausible_idea(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "PLAUSIBLE")

    result = check_for_irreconcilable_concepts("federated learning for fraud detection")

    assert result.flagged is False


def test_returns_flagged_with_reason_for_an_absurd_idea(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(
        monkeypatch, "REQUIRES_REVIEW: stacks unrelated geometry and cryptography terms with no precedent"
    )

    result = check_for_irreconcilable_concepts(
        "non-Euclidean hyper-dimensional temporal graph networks for zero-knowledge proofs"
    )

    assert result.flagged is True
    assert result.reason is not None


def test_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    bad_response = Mock()
    bad_response.json.return_value = {"message": {"content": "not a valid response"}}
    bad_response.raise_for_status = Mock()
    good_response = Mock()
    good_response.json.return_value = {"message": {"content": "PLAUSIBLE"}}
    good_response.raise_for_status = Mock()
    mock_post = Mock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("researchbridge.assessment.absurdity.requests.post", mock_post)

    result = check_for_irreconcilable_concepts("idea")

    assert result.flagged is False
    assert mock_post.call_count == 2


def test_raises_unavailable_after_two_failed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "not a valid response")

    with pytest.raises(AbsurdityCheckUnavailable, match="could not produce"):
        check_for_irreconcilable_concepts("idea")
