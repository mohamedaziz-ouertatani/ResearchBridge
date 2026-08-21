from __future__ import annotations

from researchbridge.benchmark.fulltext import _tidy


def test_control_characters_from_broken_math_fonts_are_stripped() -> None:
    # \x11 and \x00 are the kind of raw control bytes a symbol-font glyph
    # (e.g. a tensor-product operator) decodes to when a text extractor has
    # no Unicode mapping for it - not real content, just noise.
    garbled = "x⊗k\x11 and \x00d+p−1 remain readable"

    tidied = _tidy(garbled)

    assert "\x11" not in tidied
    assert "\x00" not in tidied
    assert "x⊗k" in tidied
    assert "d+p−1" in tidied


def test_private_use_area_glyphs_are_stripped() -> None:
    garbled = "beforemiddleafter"

    tidied = _tidy(garbled)

    assert "" not in tidied
    assert "" not in tidied
    assert "beforemiddleafter" in tidied


def test_supplementary_private_use_planes_are_stripped() -> None:
    garbled = "a\U000f0000b\U0010fffdc"

    tidied = _tidy(garbled)

    assert "a" in tidied and "b" in tidied and "c" in tidied
    assert "\U000f0000" not in tidied
    assert "\U0010fffd" not in tidied


def test_replacement_character_is_stripped() -> None:
    assert "�" not in _tidy("broken�glyph")


def test_genuine_math_unicode_survives() -> None:
    # These are real, correctly-decoded symbols - stripping must not touch them.
    text = "≤λ⋆ and Ω(d²) and ∑ and ∫ and p < .001"

    tidied = _tidy(text)

    assert tidied == text.strip()


def test_tab_and_newline_survive_as_before() -> None:
    text = "line one\nline two\tindented"

    assert _tidy(text) == text


def test_whitespace_collapsing_still_works_alongside_stripping() -> None:
    garbled = "para one  \n\n\n\npara two\x11 continues"

    tidied = _tidy(garbled)

    assert "\n\n\n" not in tidied
    assert "\x11" not in tidied
