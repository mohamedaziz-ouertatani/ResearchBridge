from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from researchbridge.assessment.external_validation import assess_external_validation
from researchbridge.connectors.epo_patents import EPOPatentConnectorError, PatentHit
from researchbridge.connectors.world_bank import IndicatorValue, WorldBankConnectorError


def _patent_hit() -> PatentHit:
    return PatentHit(
        title="Automated irrigation control system",
        publication_number="TN12345B1",
        applicant="Example Agritech SARL",
        publication_date=date(2023, 4, 11),
        url="https://register.epo.org/application?number=TN12345B1",
    )


def _indicator() -> IndicatorValue:
    return IndicatorValue(indicator_name="GDP", value=46255510000.0, year=2023, unit="current US$")


def test_never_asserts_a_market_potential_number_or_level() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = [_patent_hit()]
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = [_indicator()]

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    lowered = result.text.lower()
    assert "high market potential" not in lowered
    assert "strong demand" not in lowered


def test_both_sources_succeed_includes_both_sections() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = [_patent_hit()]
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = [_indicator()]

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert "TN12345B1" in result.text
    assert "Automated irrigation control system" in result.text
    assert "GDP" in result.text
    assert result.patent_hits == [_patent_hit()]
    assert result.indicator_values == [_indicator()]


def test_patent_connector_none_falls_back_for_patents_only() -> None:
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = [_indicator()]

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=None,
        market_connector=market_connector,
    )

    assert "patent" in result.text.lower()
    assert "GDP" in result.text
    assert result.patent_hits == []


def test_patent_search_error_falls_back_for_patents_only() -> None:
    patent_connector = MagicMock()
    patent_connector.search.side_effect = EPOPatentConnectorError("boom")
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = [_indicator()]

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert "GDP" in result.text
    assert result.patent_hits == []


def test_market_fetch_error_falls_back_for_market_only() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = [_patent_hit()]
    market_connector = MagicMock()
    market_connector.fetch_indicators.side_effect = WorldBankConnectorError("boom")

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert "TN12345B1" in result.text
    assert result.indicator_values == []


def test_empty_search_results_produce_found_nothing_message_not_error_message() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = []
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = []

    result = assess_external_validation(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert "no matching patent" in result.text.lower()
    patent_connector.search.assert_called_once()


def test_empty_keywords_short_circuits_without_calling_either_connector() -> None:
    patent_connector = MagicMock()
    market_connector = MagicMock()

    result = assess_external_validation(
        title=None,
        raw_text="!!! ... ---",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    patent_connector.search.assert_not_called()
    market_connector.fetch_indicators.assert_not_called()
    assert "not assessed" in result.text.lower() or "NOT ASSESSED" in result.text


def test_references_applications_when_present() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = []
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = []

    result = assess_external_validation(
        title=None,
        raw_text="!!! ... ---",
        has_applications=True,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert "application" in result.text.lower()


def test_is_deterministic_given_the_same_connector_results() -> None:
    patent_connector = MagicMock()
    patent_connector.search.return_value = [_patent_hit()]
    market_connector = MagicMock()
    market_connector.fetch_indicators.return_value = [_indicator()]

    args = dict(
        title="Irrigation idea",
        raw_text="soil moisture sensor network for automated irrigation",
        has_applications=False,
        patent_connector=patent_connector,
        market_connector=market_connector,
    )

    assert assess_external_validation(**args).text == assess_external_validation(**args).text
