from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest

from researchbridge.assessment.opportunity_synthesis import (
    OpportunitySynthesisUnavailable,
    SourceApplication,
    build_prompt,
    ollama_enabled,
    parse_response,
    synthesize_opportunities,
)


def _app(text: str, source_paper: str = "Some Paper") -> SourceApplication:
    return SourceApplication(application=text, source_paper=source_paper, paper_id=str(uuid.uuid4()))


# --- build_prompt -----------------------------------------------------------


def test_build_prompt_numbers_applications_in_order() -> None:
    apps = [_app("real-time fraud screening", "Paper A"), _app("credit scoring", "Paper B")]

    system_prompt, user_prompt = build_prompt(apps)

    assert "Direct" in system_prompt and "Adjacent" in system_prompt and "Speculative" in system_prompt
    assert '[1] "real-time fraud screening" — Paper A' in user_prompt
    assert '[2] "credit scoring" — Paper B' in user_prompt


def test_build_prompt_escapes_bracketed_numbers_already_in_application_text() -> None:
    apps = [_app("cites prior work [3] directly")]

    _, user_prompt = build_prompt(apps)

    # the literal "[3]" from the application's own text must not look like
    # a citation marker once numbered - it should be neutralized to "(3)"
    assert "[3]" not in user_prompt.split("\n", 1)[1]
    assert "(3)" in user_prompt


def test_build_prompt_states_the_valid_citation_range_and_does_not_imply_two_citations() -> None:
    # regression guard: an earlier prompt's "Adjacent: <opportunity> [n][m]"
    # example hardcoded a two-citation shape and the real model pattern-
    # matched it literally even with only one application available,
    # hallucinating a citation that didn't exist - every single-application
    # synthesis failed as a result. The prompt must state the real range
    # and must not show a two-bracket example for any one tier.
    system_prompt, _ = build_prompt([_app("only application")])

    assert "1" in system_prompt  # the valid range is stated
    assert "[n][m]" not in system_prompt


def test_build_prompt_system_prompt_reflects_the_actual_application_count() -> None:
    single_prompt, _ = build_prompt([_app("a")])
    multi_prompt, _ = build_prompt([_app("a"), _app("b"), _app("c")])

    assert single_prompt != multi_prompt


# --- parse_response -----------------------------------------------------------


def test_parse_response_accepts_three_valid_tiers_in_order() -> None:
    text = "Direct: fraud-scoring API [1]\nAdjacent: risk platform [1][2]\nSpeculative: fraud network [2]"

    result = parse_response(text, application_count=2)

    assert [o.tier for o in result] == ["direct", "adjacent", "speculative"]
    assert result[0].opportunity == "fraud-scoring API"
    assert result[0].source_application_indices == [1]
    assert result[1].source_application_indices == [1, 2]


def test_parse_response_accepts_comma_separated_citations_in_one_bracket() -> None:
    # real model behavior (qwen2.5:3b), not a contrived shape: it writes
    # multi-citations both as "[1][2]" (separate brackets, matching the
    # prompt's own example) and as "[1,2]" (comma-separated in one
    # bracket) interchangeably - a line using the second style must not
    # silently parse as having zero citations
    text = "Direct: a [1]\nAdjacent: b [1,2]\nSpeculative: c [1, 2]"

    result = parse_response(text, application_count=2)

    assert result[1].source_application_indices == [1, 2]
    assert result[2].source_application_indices == [1, 2]
    assert result[1].opportunity == "b"
    assert result[2].opportunity == "c"


def test_parse_response_reorders_tiers_regardless_of_model_output_order() -> None:
    text = "Speculative: fraud network [1]\nDirect: fraud-scoring API [1]\nAdjacent: risk platform [1]"

    result = parse_response(text, application_count=1)

    assert [o.tier for o in result] == ["direct", "adjacent", "speculative"]


def test_parse_response_raises_on_missing_tier() -> None:
    text = "Direct: fraud-scoring API [1]\nAdjacent: risk platform [1]"

    with pytest.raises(ValueError, match="missing tier"):
        parse_response(text, application_count=1)


def test_parse_response_raises_on_duplicate_tier() -> None:
    text = "Direct: first [1]\nDirect: second [1]\nAdjacent: risk platform [1]\nSpeculative: network [1]"

    with pytest.raises(ValueError, match="duplicate direct"):
        parse_response(text, application_count=1)


def test_parse_response_raises_on_out_of_range_citation() -> None:
    text = "Direct: fraud-scoring API [5]\nAdjacent: risk platform [1]\nSpeculative: network [1]"

    with pytest.raises(ValueError, match="out of range"):
        parse_response(text, application_count=1)


def test_parse_response_raises_when_a_tier_has_no_citation() -> None:
    text = "Direct: fraud-scoring API\nAdjacent: risk platform [1]\nSpeculative: network [1]"

    with pytest.raises(ValueError, match="direct opportunity has no citation"):
        parse_response(text, application_count=1)


def test_parse_response_raises_when_a_tier_has_no_text_beyond_its_citation() -> None:
    text = "Direct: [1]\nAdjacent: risk platform [1]\nSpeculative: network [1]"

    with pytest.raises(ValueError, match="no text beyond its citation"):
        parse_response(text, application_count=1)


def test_parse_response_ignores_unrelated_lines() -> None:
    text = (
        "Here are three opportunities:\n"
        "Direct: fraud-scoring API [1]\n"
        "Adjacent: risk platform [1]\n"
        "Speculative: network [1]\n"
        "Let me know if you'd like more detail."
    )

    result = parse_response(text, application_count=1)

    assert len(result) == 3


# --- synthesize_opportunities (Ollama call mocked) --------------------------


def _mock_ollama_response(monkeypatch: pytest.MonkeyPatch, content: str) -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": content}}
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)
    return mock_post


def test_ollama_enabled_reflects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    assert ollama_enabled() is True

    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    assert ollama_enabled() is False

    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    assert ollama_enabled() is False


def test_synthesize_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")

    with pytest.raises(OpportunitySynthesisUnavailable, match="not enabled"):
        synthesize_opportunities([_app("fraud screening")])


def test_synthesize_raises_when_no_applications_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")

    with pytest.raises(OpportunitySynthesisUnavailable, match="no applications"):
        synthesize_opportunities([])


def test_synthesize_returns_validated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    apps = [_app("fraud screening"), _app("credit scoring")]
    _mock_ollama_response(
        monkeypatch,
        "Direct: fraud-scoring API [1]\nAdjacent: risk platform [1][2]\nSpeculative: fraud network [2]",
    )

    result = synthesize_opportunities(apps)

    assert [o.tier for o in result.opportunities] == ["direct", "adjacent", "speculative"]


def test_synthesize_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    apps = [_app("fraud screening")]
    bad_response = Mock()
    bad_response.json.return_value = {"message": {"content": "not a valid response"}}
    bad_response.raise_for_status = Mock()
    good_response = Mock()
    good_response.json.return_value = {
        "message": {"content": "Direct: a [1]\nAdjacent: b [1]\nSpeculative: c [1]"}
    }
    good_response.raise_for_status = Mock()
    mock_post = Mock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)

    result = synthesize_opportunities(apps)

    assert len(result.opportunities) == 3
    assert mock_post.call_count == 2


def test_synthesize_fails_closed_after_two_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "not a valid response")

    with pytest.raises(OpportunitySynthesisUnavailable, match="valid grounded synthesis"):
        synthesize_opportunities([_app("fraud screening")])


def test_synthesize_fails_closed_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_post = Mock(side_effect=requests.ConnectionError("connection refused"))
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)

    with pytest.raises(OpportunitySynthesisUnavailable, match="valid grounded synthesis"):
        synthesize_opportunities([_app("fraud screening")])

    assert mock_post.call_count == 2
