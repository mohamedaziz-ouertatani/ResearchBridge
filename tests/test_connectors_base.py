from researchbridge.connectors.base import clean_harvested_abstract


def test_clean_harvested_abstract_strips_duplicated_mathml_block() -> None:
    raw = (
        r"the accuracy went from $$99.88\%\!\pm \!0.22\%$$ 99.88 % ± 0.22 % to "
        r"$$10.34\%\!\pm \!0.03\%$$ 10.34 % ± 0.03 % while maintaining a test "
        r"accuracy of $$98.93\%\!\pm \!0.03\%$$ 98.93 % ± 0.03 % , closely "
        r"matching the clean FL performance."
    )

    cleaned = clean_harvested_abstract(raw)

    assert "$$" not in cleaned
    assert cleaned == (
        "the accuracy went from 99.88 % ± 0.22 % to 10.34 % ± 0.03 % "
        "while maintaining a test accuracy of 98.93 % ± 0.03 % , closely "
        "matching the clean FL performance."
    )


def test_clean_harvested_abstract_passes_through_plain_text() -> None:
    assert (
        clean_harvested_abstract("A plain abstract with no math markup.")
        == "A plain abstract with no math markup."
    )


def test_clean_harvested_abstract_handles_none_and_empty() -> None:
    assert clean_harvested_abstract(None) is None
    assert clean_harvested_abstract("") is None
