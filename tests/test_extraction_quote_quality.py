from __future__ import annotations

from researchbridge.extraction.quote_quality import is_acceptable_quote


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
