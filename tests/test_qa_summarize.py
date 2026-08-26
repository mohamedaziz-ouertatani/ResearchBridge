from __future__ import annotations

import uuid

import pytest

from researchbridge.api.schemas import QuoteHitOut
from researchbridge.qa.summarize import build_prompt, extract_citations


def _hit(text: str, paper_title: str = "Some Paper") -> QuoteHitOut:
    return QuoteHitOut(
        paper_id=uuid.uuid4(),
        paper_title=paper_title,
        paper_source="arxiv",
        claim_type="limitations",
        text=text,
        section=None,
        confidence="medium",
        score=0.9,
    )


def test_build_prompt_numbers_hits_in_order() -> None:
    hits = [_hit("first quote", "Paper A"), _hit("second quote", "Paper B")]

    system_prompt, user_prompt = build_prompt("what are the limitations?", hits)

    assert "only use" in system_prompt.lower() or "only" in system_prompt.lower()
    assert '[1] "first quote" — Paper A' in user_prompt
    assert '[2] "second quote" — Paper B' in user_prompt
    assert "what are the limitations?" in user_prompt


def test_build_prompt_escapes_bracketed_numbers_inside_quote_text() -> None:
    hits = [_hit("prior work showed this already [5]", "Paper A")]

    _system_prompt, user_prompt = build_prompt("a question", hits)

    assert "[5]" not in user_prompt
    assert "(5)" in user_prompt


def test_extract_citations_returns_unique_numbers_in_order_of_appearance() -> None:
    text = "Models struggle offline [2]. This was also noted elsewhere [1][2]."

    citations = extract_citations(text, hit_count=2)

    assert citations == [2, 1]


def test_extract_citations_returns_empty_list_when_no_citations_present() -> None:
    citations = extract_citations("The quotes don't address this question.", hit_count=3)

    assert citations == []


def test_extract_citations_raises_on_out_of_range_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("This claims something [5].", hit_count=2)


def test_extract_citations_raises_on_zero_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("Cites [0] which isn't a valid index.", hit_count=2)


from unittest.mock import Mock

from researchbridge.qa.summarize import (
    SummarizationUnavailable,
    ollama_enabled,
    summarize_quotes,
)


def _mock_ollama_response(monkeypatch: pytest.MonkeyPatch, content: str) -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": content}}
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)
    return mock_post


def test_ollama_enabled_reflects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    assert ollama_enabled() is True

    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    assert ollama_enabled() is False

    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    assert ollama_enabled() is False


def test_summarize_quotes_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    hits = [_hit("some quote")]

    with pytest.raises(SummarizationUnavailable, match="not enabled"):
        summarize_quotes("a question", hits)


def test_summarize_quotes_returns_validated_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("first quote"), _hit("second quote")]
    _mock_ollama_response(monkeypatch, "This is grounded [1] and also this [2].")

    result = summarize_quotes("a question", hits)

    assert result.summary == "This is grounded [1] and also this [2]."
    assert result.citations == [1, 2]


def test_summarize_quotes_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    mock_response_bad = Mock()
    mock_response_bad.json.return_value = {"message": {"content": "Invalid cite [9]."}}
    mock_response_bad.raise_for_status = Mock()
    mock_response_good = Mock()
    mock_response_good.json.return_value = {"message": {"content": "Valid cite [1]."}}
    mock_response_good.raise_for_status = Mock()
    mock_post = Mock(side_effect=[mock_response_bad, mock_response_good])
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)

    result = summarize_quotes("a question", hits)

    assert result.summary == "Valid cite [1]."
    assert mock_post.call_count == 2


def test_summarize_quotes_fails_closed_after_two_bad_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    _mock_ollama_response(monkeypatch, "Always invalid [9].")

    with pytest.raises(SummarizationUnavailable, match="valid grounded summary"):
        summarize_quotes("a question", hits)


def test_summarize_quotes_fails_closed_when_ollama_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    mock_post = Mock(side_effect=requests.ConnectionError("connection refused"))
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)

    with pytest.raises(SummarizationUnavailable, match="valid grounded summary"):
        summarize_quotes("a question", hits)

    assert mock_post.call_count == 2
