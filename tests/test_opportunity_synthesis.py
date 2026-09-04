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


# Filler opportunity text used throughout below, deliberately >=16
# characters (MIN_OPPORTUNITY_TEXT_LENGTH) so these tests exercise their
# own stated concern (formatting/citation parsing) without incidentally
# tripping the separate content-length validation - see
# test_parse_response_raises_when_a_tier_is_too_short below for that
# check's own dedicated tests.
ALPHA = "alpha opportunity"
BETA = "beta opportunity"
GAMMA = "gamma opportunity"


# --- build_prompt -----------------------------------------------------------


def test_build_prompt_numbers_applications_in_order() -> None:
    apps = [_app("real-time fraud screening", "Paper A"), _app("credit scoring", "Paper B")]

    system_prompt, user_prompt = build_prompt("a fraud detection idea", apps)

    assert "Direct" in system_prompt and "Adjacent" in system_prompt and "Speculative" in system_prompt
    assert '[1] "real-time fraud screening" — Paper A' in user_prompt
    assert '[2] "credit scoring" — Paper B' in user_prompt


def test_build_prompt_includes_the_idea_text() -> None:
    # 2026-09-04 regression guard: a real live failure showed the model
    # free-associating across an application's OWN unrelated enumerated
    # examples ("used in X, Y, Z, traffic prediction") into wildly
    # off-topic opportunities (healthcare/finance for a traffic idea) -
    # traced to the prompt never telling the model what the idea WAS at
    # all, only showing it the applications. The idea must actually be
    # in the prompt now, and the system prompt must instruct staying
    # anchored to it when an application lists multiple examples.
    _, user_prompt = build_prompt("a traffic congestion prediction idea", [_app("an application")])

    assert "a traffic congestion prediction idea" in user_prompt


def test_build_prompt_system_prompt_warns_against_unrelated_enumerated_examples() -> None:
    system_prompt, _ = build_prompt("an idea", [_app("an application")])

    assert "unrelated" in system_prompt.lower()


def test_build_prompt_escapes_bracketed_numbers_already_in_application_text() -> None:
    apps = [_app("cites prior work [3] directly")]

    _, user_prompt = build_prompt("an idea", apps)

    # the literal "[3]" from the application's own text must not look like
    # a citation marker once numbered - it should be neutralized to "(3)"
    assert "[3]" not in user_prompt.split("Applications:\n", 1)[1]
    assert "(3)" in user_prompt


def test_build_prompt_escapes_bracketed_numbers_already_in_the_idea_text() -> None:
    _, user_prompt = build_prompt("an idea citing prior work [3] directly", [_app("an application")])

    assert "[3]" not in user_prompt.split("Applications:\n", 1)[0]
    assert "(3)" in user_prompt


def test_build_prompt_states_the_valid_citation_range_and_does_not_imply_two_citations() -> None:
    # regression guard: an earlier prompt's "Adjacent: <opportunity> [n][m]"
    # example hardcoded a two-citation shape and the real model pattern-
    # matched it literally even with only one application available,
    # hallucinating a citation that didn't exist - every single-application
    # synthesis failed as a result. The prompt must state the real range
    # and must not show a two-bracket example for any one tier.
    system_prompt, _ = build_prompt("an idea", [_app("only application")])

    assert "1" in system_prompt  # the valid range is stated
    assert "[n][m]" not in system_prompt


def test_build_prompt_system_prompt_reflects_the_actual_application_count() -> None:
    single_prompt, _ = build_prompt("an idea", [_app("a")])
    multi_prompt, _ = build_prompt("an idea", [_app("a"), _app("b"), _app("c")])

    assert single_prompt != multi_prompt


# --- parse_response -----------------------------------------------------------


def test_parse_response_accepts_three_valid_tiers_in_order() -> None:
    text = "Direct: fraud-scoring API [1]\nAdjacent: fraud risk platform [1][2]\nSpeculative: fraud detection network [2]"

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
    text = f"Direct: {ALPHA} [1]\nAdjacent: {BETA} [1,2]\nSpeculative: {GAMMA} [1, 2]"

    result = parse_response(text, application_count=2)

    assert result[1].source_application_indices == [1, 2]
    assert result[2].source_application_indices == [1, 2]
    assert result[1].opportunity == BETA
    assert result[2].opportunity == GAMMA


def test_parse_response_reorders_tiers_regardless_of_model_output_order() -> None:
    text = (
        "Speculative: fraud detection network [1]\n"
        "Direct: fraud-scoring API [1]\n"
        "Adjacent: fraud risk platform [1]"
    )

    result = parse_response(text, application_count=1)

    assert [o.tier for o in result] == ["direct", "adjacent", "speculative"]


def test_parse_response_raises_on_missing_tier() -> None:
    text = "Direct: fraud-scoring API [1]\nAdjacent: fraud risk platform [1]"

    with pytest.raises(ValueError, match="missing tier"):
        parse_response(text, application_count=1)


def test_parse_response_raises_on_duplicate_tier() -> None:
    text = (
        "Direct: first opportunity idea [1]\n"
        "Direct: second opportunity idea [1]\n"
        "Adjacent: fraud risk platform [1]\n"
        "Speculative: fraud detection network [1]"
    )

    with pytest.raises(ValueError, match="duplicate direct"):
        parse_response(text, application_count=1)


def test_parse_response_raises_on_out_of_range_citation() -> None:
    text = "Direct: fraud-scoring API [5]\nAdjacent: fraud risk platform [1]\nSpeculative: fraud detection network [1]"

    with pytest.raises(ValueError, match="out of range"):
        parse_response(text, application_count=1)


def test_parse_response_raises_when_a_tier_has_no_citation() -> None:
    text = "Direct: fraud-scoring API\nAdjacent: fraud risk platform [1]\nSpeculative: fraud detection network [1]"

    with pytest.raises(ValueError, match="direct opportunity has no citation"):
        parse_response(text, application_count=1)


def test_parse_response_raises_when_a_tier_has_no_text_beyond_its_citation() -> None:
    text = "Direct: [1]\nAdjacent: fraud risk platform [1]\nSpeculative: fraud detection network [1]"

    with pytest.raises(ValueError, match="no text beyond its citation"):
        parse_response(text, application_count=1)


def test_parse_response_ignores_unrelated_lines() -> None:
    text = (
        "Here are three opportunities:\n"
        "Direct: fraud-scoring API [1]\n"
        "Adjacent: fraud risk platform [1]\n"
        "Speculative: fraud detection network [1]\n"
        "Let me know if you'd like more detail."
    )

    result = parse_response(text, application_count=1)

    assert len(result) == 3


# --- content-quality validation (MIN_OPPORTUNITY_TEXT_LENGTH) ---------------
# Item 8 of the assessment hardening list: systematic cross-model
# verification surfaced a real, repeatedly-reproducible failure mode in the
# DEFAULT model specifically (qwen2.5:3b, not the other 3 locally-available
# models tested) - a structurally perfect line whose "opportunity" is a bare
# category word ("Evaluate", "Metrics", "Scale"), which the format/citation
# checks above all happily accept.


def test_parse_response_raises_when_a_tier_is_too_short() -> None:
    # real, repeatedly-reproduced output from the default model
    # (qwen2.5:3b) on a single-application case: syntactically valid,
    # semantically empty
    text = "Direct: Evaluate [1]\nAdjacent: Compare [1]\nSpeculative: Scale [1]"

    with pytest.raises(ValueError, match="too short to be a real opportunity"):
        parse_response(text, application_count=1)


def test_parse_response_accepts_a_short_but_real_squished_product_name() -> None:
    # non-regression: a genuine single-token product name (no spaces) must
    # not be penalized just because it has no internal word boundary - a
    # character-length check, not a word-count one, was chosen specifically
    # to avoid this false rejection (also real default-model output, seen
    # on a different case)
    text = "Direct: HealthWellnessPlatform [1]\nAdjacent: fraud risk platform [1]\nSpeculative: fraud detection network [1]"

    result = parse_response(text, application_count=1)

    assert result[0].opportunity == "HealthWellnessPlatform"


# --- formatting robustness (_normalize_tier_line) ----------------------------
# 2026-09-03 stress-testing pass, see docs/superpowers/specs/
# 2026-09-03-opportunities-synthesis-design.md: found live that 3/3 tested
# models write plain "Direct: ..." with no markdown or numbering, so none
# of these are observed failures - forward-hardening in case a future
# OLLAMA_MODEL swap writes differently. Formatting-only: none of these
# change what counts as a valid tier word, citation, or range.


def test_parse_response_accepts_plain_baseline() -> None:
    text = f"Direct: {ALPHA} [1]\nAdjacent: {BETA} [1]\nSpeculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_accepts_markdown_bold_labels() -> None:
    text = f"**Direct:** {ALPHA} [1]\n**Adjacent:** {BETA} [1]\n**Speculative:** {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_accepts_numbered_dot_list() -> None:
    text = f"1. Direct: {ALPHA} [1]\n2. Adjacent: {BETA} [1]\n3. Speculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_accepts_numbered_parenthesis_list() -> None:
    text = f"1) Direct: {ALPHA} [1]\n2) Adjacent: {BETA} [1]\n3) Speculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_accepts_numbered_and_markdown_combined() -> None:
    text = f"1. **Direct:** {ALPHA} [1]\n2. **Adjacent:** {BETA} [1]\n3. **Speculative:** {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_accepts_mixed_formatting_across_tiers() -> None:
    # each tier written in a different style in the same response
    text = f"1. Direct: {ALPHA} [1]\n**Adjacent:** {BETA} [1]\n- Speculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


# --- preserved regression coverage: normalization must not loosen validation


def test_parse_response_still_handles_eight_applications() -> None:
    text = f"Direct: {ALPHA} [7]\nAdjacent: {BETA} [7,8]\nSpeculative: {GAMMA} [1,2,3,4,5,6,7,8]"

    result = parse_response(text, application_count=8)

    assert result[0].source_application_indices == [7]
    assert result[1].source_application_indices == [7, 8]
    assert result[2].source_application_indices == [1, 2, 3, 4, 5, 6, 7, 8]


def test_parse_response_dedupes_repeated_citation_in_one_bracket() -> None:
    text = f"Direct: {ALPHA} [1,1]\nAdjacent: {BETA} [1]\nSpeculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert result[0].source_application_indices == [1]


def test_parse_response_dedupes_repeated_citation_across_separate_brackets() -> None:
    text = f"Direct: {ALPHA} [1][1]\nAdjacent: {BETA} [1]\nSpeculative: {GAMMA} [1]"

    result = parse_response(text, application_count=1)

    assert result[0].source_application_indices == [1]


def test_parse_response_still_raises_on_out_of_range_citation_with_markdown() -> None:
    # normalization must not accidentally widen the valid citation range
    text = f"**Direct:** {ALPHA} [5]\n**Adjacent:** {BETA} [1]\n**Speculative:** {GAMMA} [1]"

    with pytest.raises(ValueError, match="out of range"):
        parse_response(text, application_count=1)


def test_parse_response_still_raises_on_missing_tier_with_numbered_formatting() -> None:
    text = f"1. Direct: {ALPHA} [1]\n2. Adjacent: {BETA} [1]"

    with pytest.raises(ValueError, match="missing tier"):
        parse_response(text, application_count=1)


def test_parse_response_still_rejects_malformed_synonym_tier_names() -> None:
    # normalization strips list/markdown noise only - it must never make a
    # tier word the model was never asked to use suddenly recognized, even
    # if it reads as a plausible synonym ("Indirect" for Adjacent,
    # "Long-term" for Speculative)
    text = f"Direct: {ALPHA} [1]\nIndirect: {BETA} [1]\nLong-term: {GAMMA} [1]"

    with pytest.raises(ValueError, match="missing tier"):
        parse_response(text, application_count=1)


def test_parse_response_still_ignores_irrelevant_prose_with_list_markers() -> None:
    # a bullet-prefixed prose line must not be mistaken for a tier line
    # just because normalization strips its leading "- "
    text = (
        "- Here is my analysis:\n"
        f"Direct: {ALPHA} [1]\n"
        f"Adjacent: {BETA} [1]\n"
        f"Speculative: {GAMMA} [1]\n"
        "- Hope this helps!"
    )

    result = parse_response(text, application_count=1)

    assert [o.opportunity for o in result] == [ALPHA, BETA, GAMMA]


def test_parse_response_still_raises_when_a_markdown_wrapped_tier_has_no_citation() -> None:
    text = f"**Direct:** {ALPHA}\n**Adjacent:** {BETA} [1]\n**Speculative:** {GAMMA} [1]"

    with pytest.raises(ValueError, match="direct opportunity has no citation"):
        parse_response(text, application_count=1)


def test_parse_response_still_raises_on_duplicate_tier_with_numbered_formatting() -> None:
    text = f"1. Direct: {ALPHA} [1]\n2. Direct: {BETA} [1]\n3. Adjacent: {GAMMA} [1]\n4. Speculative: {ALPHA} [1]"

    with pytest.raises(ValueError, match="duplicate direct"):
        parse_response(text, application_count=1)


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
        synthesize_opportunities("an idea", [_app("fraud screening")])


def test_synthesize_raises_when_no_applications_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")

    with pytest.raises(OpportunitySynthesisUnavailable, match="no applications"):
        synthesize_opportunities("an idea", [])


def test_synthesize_returns_validated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    apps = [_app("fraud screening"), _app("credit scoring")]
    _mock_ollama_response(
        monkeypatch,
        "Direct: fraud-scoring API [1]\nAdjacent: fraud risk platform [1][2]\nSpeculative: fraud detection network [2]",
    )

    result = synthesize_opportunities("an idea", apps)

    assert [o.tier for o in result.opportunities] == ["direct", "adjacent", "speculative"]


def test_synthesize_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    apps = [_app("fraud screening")]
    bad_response = Mock()
    bad_response.json.return_value = {"message": {"content": "not a valid response"}}
    bad_response.raise_for_status = Mock()
    good_response = Mock()
    good_response.json.return_value = {
        "message": {"content": f"Direct: {ALPHA} [1]\nAdjacent: {BETA} [1]\nSpeculative: {GAMMA} [1]"}
    }
    good_response.raise_for_status = Mock()
    mock_post = Mock(side_effect=[bad_response, good_response])
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)

    result = synthesize_opportunities("an idea", apps)

    assert len(result.opportunities) == 3
    assert mock_post.call_count == 2


def test_synthesize_retries_once_after_a_too_short_response_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # the content-length check must feed the same retry-then-fail-closed
    # path as every other validation failure, not a special case
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    apps = [_app("fraud screening")]
    degenerate_response = Mock()
    degenerate_response.json.return_value = {
        "message": {"content": "Direct: Evaluate [1]\nAdjacent: Compare [1]\nSpeculative: Scale [1]"}
    }
    degenerate_response.raise_for_status = Mock()
    good_response = Mock()
    good_response.json.return_value = {
        "message": {"content": f"Direct: {ALPHA} [1]\nAdjacent: {BETA} [1]\nSpeculative: {GAMMA} [1]"}
    }
    good_response.raise_for_status = Mock()
    mock_post = Mock(side_effect=[degenerate_response, good_response])
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)

    result = synthesize_opportunities("an idea", apps)

    assert [o.opportunity for o in result.opportunities] == [ALPHA, BETA, GAMMA]
    assert mock_post.call_count == 2


def test_synthesize_fails_closed_after_two_invalid_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama_response(monkeypatch, "not a valid response")

    with pytest.raises(OpportunitySynthesisUnavailable, match="valid grounded synthesis"):
        synthesize_opportunities("an idea", [_app("fraud screening")])


def test_synthesize_fails_closed_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_post = Mock(side_effect=requests.ConnectionError("connection refused"))
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)

    with pytest.raises(OpportunitySynthesisUnavailable, match="valid grounded synthesis"):
        synthesize_opportunities("an idea", [_app("fraud screening")])

    assert mock_post.call_count == 2
