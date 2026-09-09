from __future__ import annotations

from researchbridge.extraction.quote_quality import is_acceptable_quote, looks_like_heading, looks_truncated


def test_rejects_bare_word_fragment() -> None:
    assert not is_acceptable_quote("For")


def test_rejects_hyphenated_line_wrap_cutoff() -> None:
    assert not is_acceptable_quote("We propose DU-")


def test_rejects_mid_sentence_cutoff_no_terminal_punctuation() -> None:
    assert not is_acceptable_quote("However, CWE-022 and CWE-295 ship no")


def test_rejects_copyright_notice() -> None:
    assert not is_acceptable_quote("© 2020, Springer Nature Switzerland AG.")


def test_rejects_publisher_copyright_line() -> None:
    assert not is_acceptable_quote("Publisher Copyright: © 2024 The Author(s).")


def test_rejects_correspondence_email_block() -> None:
    assert not is_acceptable_quote(
        "Correspondence to: Jiangchao Yao <sunarker@sjtu.edu.cn>, Shuai Xiao <shuai.xsh@gmail.com>."
    )


def test_rejects_bare_section_heading() -> None:
    assert not is_acceptable_quote("Sequential Consistency of Attention Heads")


def test_rejects_bare_numbered_section_heading() -> None:
    assert not is_acceptable_quote("Discussion, Limitations, and Future Work 8.1.")


def test_accepts_genuine_complete_sentence() -> None:
    assert is_acceptable_quote(
        "We propose AutoSLO, a learning-based, self-adaptive scaling framework that "
        "dynamically adjusts microservice replicas to meet SLOs while minimizing resource usage."
    )


def test_accepts_short_but_complete_technical_sentence() -> None:
    assert is_acceptable_quote("Static analyzers have been widely adopted for vulnerability detection.")


def test_rejects_quote_cut_off_at_a_comparison_abbreviation() -> None:
    """A trailing "vs." is an abbreviation, not a sentence end, so the
    quote is a fragment even though it ends in a period. Found live
    2026-09-09 in a real report: "However, YOLO26 achieved significantly
    superior specificity (96.1% vs." was rendered as a risk."""
    assert not is_acceptable_quote(
        "However, YOLO26 achieved significantly superior specificity (96.1% vs."
    )


def test_rejects_quote_cut_off_at_a_citation_abbreviation() -> None:
    assert not is_acceptable_quote("Results were compared against the baselines of Smith et al.")


def test_rejects_quote_with_an_unclosed_parenthesis() -> None:
    assert not is_acceptable_quote(
        "The system was evaluated on three public benchmarks (MIMIC-CXR, CheXpert and"
    )


def test_accepts_a_sentence_with_balanced_parentheses() -> None:
    assert is_acceptable_quote(
        "The system was evaluated on three public benchmarks (MIMIC-CXR, CheXpert, PadChest)."
    )


def test_accepts_a_sentence_containing_an_abbreviation_that_is_not_at_the_end() -> None:
    assert is_acceptable_quote(
        "We compare our approach vs. the strongest published baseline on every dataset."
    )


def test_looks_truncated_flags_a_trailing_abbreviation() -> None:
    assert looks_truncated("However, YOLO26 achieved significantly superior specificity (96.1% vs.")


def test_looks_truncated_flags_an_unclosed_parenthesis() -> None:
    assert looks_truncated("It was evaluated on three benchmarks (MIMIC-CXR, CheXpert and")


def test_looks_truncated_flags_a_hyphenated_line_wrap() -> None:
    assert looks_truncated("We propose DU-")


def test_missing_sentence_end_is_caught_at_extraction_not_at_render() -> None:
    # REVISED 2026-09-09: looks_truncated deliberately stopped treating
    # missing terminal punctuation as a cut, because at render time that
    # rule discards complete sentences whose final period was lost in
    # extraction (15.1% of stored research_gap claims). is_acceptable_quote
    # keeps the rule for its own extraction-time purpose.
    assert not looks_truncated("However, CWE-022 and CWE-295 ship no")
    assert not is_acceptable_quote("However, CWE-022 and CWE-295 ship no")


def test_looks_truncated_accepts_a_short_but_complete_sentence() -> None:
    """Unlike is_acceptable_quote, this predicate is for text that already
    passed the extraction gate, so it must not re-impose a length floor."""
    assert not looks_truncated("Problem A text.")


def test_looks_truncated_accepts_an_ordinary_sentence() -> None:
    assert not looks_truncated("Manual interpretation of chest X-rays is slow and error-prone.")


def test_looks_like_heading_flags_a_bare_section_heading() -> None:
    """Found live 2026-09-09: an assessment reported its research gap as
    "Limitations and Future Work / Limited Distance Metrics." - two stacked
    section headings, no gap statement. is_acceptable_quote missed it
    because lowercase function words ("and") pulled the capitalized-word
    ratio under its 0.8 bar."""
    assert looks_like_heading("Limitations and Future Work.")
    assert looks_like_heading("Limitations and Future Work\nLimited Distance Metrics.")


def test_looks_like_heading_flags_a_numbered_heading() -> None:
    assert looks_like_heading("3.2 Limitations and Future Work")


def test_looks_like_heading_accepts_a_real_sentence() -> None:
    assert not looks_like_heading(
        "No prior work evaluates these models under distribution shift at deployment time."
    )


def test_looks_like_heading_accepts_a_sentence_that_starts_with_capitals() -> None:
    assert not looks_like_heading(
        "Limitations of Existing Benchmarks are discussed by several authors in this area."
    )


def test_heading_only_text_is_not_an_acceptable_quote() -> None:
    assert not is_acceptable_quote("Limitations and Future Work.")


def test_looks_truncated_flags_a_sentence_cut_at_an_enumeration_marker() -> None:
    """Found live 2026-09-09: after heading rejection, a research gap read
    "...motivates two concrete directions for future work: (i)." - the
    sentence was cut at the first enumerated item, so the quote promises a
    list and then delivers nothing. Terminal punctuation and balanced
    parentheses meant the earlier checks all passed."""
    assert looks_truncated(
        "This structural mismatch motivates two concrete directions for future work: (i)."
    )
    assert looks_truncated("We address this in two ways: (1)")
    assert looks_truncated("The contributions are threefold: (a).")


def test_looks_truncated_flags_a_non_ascii_enumeration_marker() -> None:
    assert looks_truncated(
        "This structural mismatch motivates two concrete directions for future work: (\U0001d456)."
    )


def test_looks_truncated_accepts_a_parenthetical_that_ends_a_real_sentence() -> None:
    assert not looks_truncated(
        "We evaluate on three corpora (MIMIC-CXR, CheXpert and PadChest)."
    )


def test_looks_truncated_accepts_a_short_trailing_acronym_in_parentheses() -> None:
    assert not looks_truncated(
        "The method is evaluated against retrieval-augmented generation (RAG)."
    )


def test_looks_truncated_does_not_flag_a_sentence_merely_missing_its_final_period() -> None:
    """Render-time filtering runs over claims ALREADY stored, and 15.1% of
    this corpus's 5,370 research_gap claims are complete sentences whose
    final period was lost in extraction ("Recommendations are provided for
    future research and institutional integration"). Treating those as
    truncated discarded real signal - measured 2026-09-09. Missing
    punctuation is weak evidence of a cut; a dangling bracket or
    abbreviation is strong evidence, and only those are used here.

    is_acceptable_quote still demands terminal punctuation in its own
    right, so extraction-time behaviour is unchanged."""
    assert not looks_truncated("Recommendations are provided for future research and institutional integration")


def test_is_acceptable_quote_still_requires_terminal_punctuation() -> None:
    assert not is_acceptable_quote("However, CWE-022 and CWE-295 ship no")
