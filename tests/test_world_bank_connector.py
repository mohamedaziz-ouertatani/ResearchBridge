from __future__ import annotations

from pathlib import Path

import responses

from researchbridge.connectors.world_bank import (
    WORLD_BANK_API_BASE,
    WorldBankConnector,
    WorldBankConnectorError,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _indicator_url(code: str, country: str = "TN") -> str:
    return f"{WORLD_BANK_API_BASE}/country/{country}/indicator/{code}"


@responses.activate
def test_fetch_indicators_returns_values_for_each_configured_indicator() -> None:
    connector = WorldBankConnector()
    for code in connector.indicator_codes():
        responses.add(responses.GET, _indicator_url(code), body=_load("world_bank_gdp.json"), status=200)

    values = connector.fetch_indicators(country="TN")

    assert len(values) == len(connector.indicator_codes())
    assert values[0].value == 46255510000.0
    assert values[0].year == 2023


@responses.activate
def test_missing_indicator_data_is_silently_omitted() -> None:
    connector = WorldBankConnector()
    codes = connector.indicator_codes()
    responses.add(responses.GET, _indicator_url(codes[0]), body=_load("world_bank_no_data.json"), status=200)
    for code in codes[1:]:
        responses.add(responses.GET, _indicator_url(code), body=_load("world_bank_gdp.json"), status=200)

    values = connector.fetch_indicators(country="TN")

    assert len(values) == len(codes) - 1


@responses.activate
def test_total_request_failure_raises() -> None:
    connector = WorldBankConnector()
    for code in connector.indicator_codes():
        responses.add(responses.GET, _indicator_url(code), status=500)

    try:
        connector.fetch_indicators(country="TN")
        raise AssertionError("expected WorldBankConnectorError")
    except WorldBankConnectorError:
        pass


@responses.activate
def test_partial_failure_still_returns_successful_indicators() -> None:
    connector = WorldBankConnector()
    codes = connector.indicator_codes()
    responses.add(responses.GET, _indicator_url(codes[0]), status=500)
    for code in codes[1:]:
        responses.add(responses.GET, _indicator_url(code), body=_load("world_bank_gdp.json"), status=200)

    values = connector.fetch_indicators(country="TN")

    assert len(values) == len(codes) - 1
