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


# --- stripped-HTML-entity repair (real CORE-connector examples, verified
# against the live corpus 2026-09-03 before writing this fix) -------------


def test_clean_harvested_abstract_decodes_well_formed_entities() -> None:
    # a real CORE abstract had "&#x2019;" left completely undecoded
    raw = "This review explores CSA&#x2019;s role in mitigating risk."
    assert clean_harvested_abstract(raw) == "This review explores CSA’s role in mitigating risk."


def test_clean_harvested_abstract_repairs_stripped_curly_quotes_glued_to_words() -> None:
    # real example: "the gap between 201C;Average Daily Demand201D; and
    # 201C;Instantaneous Demand201D; of a consumer"
    raw = (
        "continuously computes the gap between 201C;Average Daily "
        "Demand201D; and 201C;Instantaneous Demand201D; of a consumer"
    )
    cleaned = clean_harvested_abstract(raw)

    assert cleaned == (
        "continuously computes the gap between “Average Daily "
        "Demand” and “Instantaneous Demand” of a consumer"
    )


def test_clean_harvested_abstract_repairs_stripped_apostrophe_glued_to_word() -> None:
    # real example: "...to incorporate end users2019; awareness and..."
    raw = "to incorporate end users2019; awareness and implications"
    assert clean_harvested_abstract(raw) == "to incorporate end users’ awareness and implications"


def test_clean_harvested_abstract_does_not_touch_a_genuine_citation_year() -> None:
    # real examples that must be left completely untouched: every one has
    # whitespace/punctuation before the year, unlike the glued corruption
    # cases above
    for raw in [
        "Recorded Oct. 7, 2019; By Dr. Joshua Kroll",
        "prior work on rate problems [Kaiser 2019; Nasim 2019], and challenges",
        "a comparison of strategies (Chui et al. 2018; Lytras et al. 2018)",
        "published in Genome Med. 2019;11:70. The application",
    ]:
        assert clean_harvested_abstract(raw) == raw


def test_clean_harvested_abstract_does_not_touch_unrelated_four_digit_numbers() -> None:
    # codes outside the verified whitelist (not observed as real
    # corruption) are left alone rather than guessed at
    raw = "grant reference 0463; project identifier 6205; see appendix"
    assert clean_harvested_abstract(raw) == raw
