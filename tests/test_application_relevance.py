from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest

from researchbridge.assessment.application_relevance import (
    ApplicationRelevanceUnavailable,
    filter_relevant_applications,
    ollama_enabled,
    parse_response,
)
from researchbridge.assessment.applications import ApplicationRecord


def _app(text: str, source_paper: str = "Some Paper") -> ApplicationRecord:
    return ApplicationRecord(application=text, source_paper=source_paper, paper_id=uuid.uuid4(), evidence_id=uuid.uuid4())


# --- parse_response -----------------------------------------------------


def test_parse_response_returns_relevant_indices() -> None:
    text = "1: relevant\n2: irrelevant\n3: relevant"

    result = parse_response(text, application_count=3)

    assert result == {1, 3}


def test_parse_response_accepts_bracketed_indices() -> None:
    text = "[1]: relevant\n[2]: irrelevant"

    result = parse_response(text, application_count=2)

    assert result == {1}


def test_parse_response_accepts_various_separators_and_leading_markers() -> None:
    text = "- 1. relevant\n* 2) irrelevant"

    result = parse_response(text, application_count=2)

    assert result == {1}


def test_parse_response_ignores_unrelated_lines() -> None:
    text = "Here are my judgments:\n1: relevant\n2: irrelevant\nThanks!"

    result = parse_response(text, application_count=2)

    assert result == {1}


def test_parse_response_raises_on_missing_index() -> None:
    text = "1: relevant"

    with pytest.raises(ValueError, match="missing judgment"):
        parse_response(text, application_count=2)


def test_parse_response_raises_on_duplicate_index() -> None:
    text = "1: relevant\n1: irrelevant\n2: relevant"

    with pytest.raises(ValueError, match="duplicate judgment"):
        parse_response(text, application_count=2)


def test_parse_response_raises_on_out_of_range_index() -> None:
    text = "1: relevant\n2: relevant\n3: relevant"

    with pytest.raises(ValueError, match="out-of-range"):
        parse_response(text, application_count=2)


def test_parse_response_all_relevant() -> None:
    text = "1: relevant\n2: relevant"

    assert parse_response(text, application_count=2) == {1, 2}


def test_parse_response_all_irrelevant() -> None:
    text = "1: irrelevant\n2: irrelevant"

    assert parse_response(text, application_count=2) == set()


# --- filter_relevant_applications (Ollama call mocked) -------------------


def _mock_ollama_response(monkeypatch: pytest.MonkeyPatch, content: str) -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": content}}
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("researchbridge.assessment.application_relevance.requests.post", mock_post)
    return mock_post


def test_ollama_enabled_reflects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    assert ollama_enabled() is True

    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    assert ollama_enabled() is False

    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    assert ollama_enabled() is False


def test_filter_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")

    with pytest.raises(ApplicationRelevanceUnavailable, match="not enabled"):
        filter_relevant_applications("some idea", [_app("an application")])


def test_filter_raises_when_no_applications_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")

    with pytest.raises(ApplicationRelevanceUnavailable, match="no applications"):
        filter_relevant_applications("some idea", [])


def test_filter_keeps_only_relevant_applications(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    keep = _app("fraud screening for federated banks")
    drop = _app("flower irrigation status tracker")
    _mock_ollama_response(monkeypatch, "1: relevant\n2: irrelevant")

    result = filter_relevant_applications("federated fraud detection idea", [keep, drop])

    assert result == [keep]


def test_filter_preserves_original_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    a, b, c = _app("app a"), _app("app b"), _app("app c")
    _mock_ollama_response(monkeypatch, "1: relevant\n2: relevant\n3: relevant")

    result = filter_relevant_applications("idea", [a, b, c])

    assert result == [a, b, c]


def test_filter_can_return_empty_list_when_none_are_relevant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "1: irrelevant")

    result = filter_relevant_applications("idea", [_app("unrelated application")])

    assert result == []


def test_filter_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    app = _app("an application")
    bad_response = Mock()
    bad_response.json.return_value = {"message": {"content": "not a valid response"}}
    bad_response.raise_for_status = Mock()
    good_response = Mock()
    good_response.json.return_value = {"message": {"content": "1: relevant"}}
    good_response.raise_for_status = Mock()
    mock_post = Mock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("researchbridge.assessment.application_relevance.requests.post", mock_post)

    result = filter_relevant_applications("idea", [app])

    assert result == [app]
    assert mock_post.call_count == 2


def test_filter_fails_open_signal_after_two_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    # "fails open" here means: raises so the CALLER can fall back to the
    # unfiltered deterministic list - see build.py's own call site, which
    # catches this and keeps applications unchanged rather than propagating
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "not a valid response")

    with pytest.raises(ApplicationRelevanceUnavailable, match="valid relevance judgment"):
        filter_relevant_applications("idea", [_app("an application")])


def test_filter_raises_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_post = Mock(side_effect=requests.ConnectionError("connection refused"))
    monkeypatch.setattr("researchbridge.assessment.application_relevance.requests.post", mock_post)

    with pytest.raises(ApplicationRelevanceUnavailable, match="valid relevance judgment"):
        filter_relevant_applications("idea", [_app("an application")])
