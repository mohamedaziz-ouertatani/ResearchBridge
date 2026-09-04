from __future__ import annotations

from pathlib import Path

import responses

from researchbridge.connectors.epo_patents import (
    EPO_OPS_SEARCH_URL,
    EPO_OPS_TOKEN_URL,
    EPOPatentConnector,
    EPOPatentConnectorError,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _mock_token() -> None:
    responses.add(
        responses.POST,
        EPO_OPS_TOKEN_URL,
        json={"access_token": "fake-token", "expires_in": "1200"},
        status=200,
    )


def test_missing_consumer_key_raises() -> None:
    try:
        EPOPatentConnector(consumer_key=None, consumer_secret="secret")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_missing_consumer_secret_raises() -> None:
    try:
        EPOPatentConnector(consumer_key="key", consumer_secret=None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


@responses.activate
def test_search_returns_normalized_patent_hits() -> None:
    _mock_token()
    responses.add(responses.GET, EPO_OPS_SEARCH_URL, body=_load("epo_ops_search_results.xml"), status=200)

    connector = EPOPatentConnector(consumer_key="key", consumer_secret="secret")
    hits = connector.search(keywords=["irrigation", "soil moisture"], country="TN")

    assert len(hits) == 1
    assert hits[0].title == "Automated irrigation control system"
    assert hits[0].publication_number == "TN12345B1"
    assert hits[0].applicant == "Example Agritech SARL"
    assert hits[0].publication_date.isoformat() == "2023-04-11"


@responses.activate
def test_search_returns_empty_list_for_no_results() -> None:
    _mock_token()
    responses.add(responses.GET, EPO_OPS_SEARCH_URL, body=_load("epo_ops_no_results.xml"), status=200)

    connector = EPOPatentConnector(consumer_key="key", consumer_secret="secret")
    hits = connector.search(keywords=["nonexistent widget"], country="TN")

    assert hits == []


@responses.activate
def test_token_fetch_failure_raises_connector_error() -> None:
    responses.add(responses.POST, EPO_OPS_TOKEN_URL, status=401)

    connector = EPOPatentConnector(consumer_key="bad", consumer_secret="bad")
    try:
        connector.search(keywords=["irrigation"], country="TN")
        raise AssertionError("expected EPOPatentConnectorError")
    except EPOPatentConnectorError:
        pass


@responses.activate
def test_search_request_failure_after_retries_raises_connector_error() -> None:
    _mock_token()
    for _ in range(3):
        responses.add(responses.GET, EPO_OPS_SEARCH_URL, status=500)

    connector = EPOPatentConnector(consumer_key="key", consumer_secret="secret")
    try:
        connector.search(keywords=["irrigation"], country="TN")
        raise AssertionError("expected EPOPatentConnectorError")
    except EPOPatentConnectorError:
        pass


@responses.activate
def test_empty_keywords_raises_value_error_without_any_request() -> None:
    connector = EPOPatentConnector(consumer_key="key", consumer_secret="secret")
    try:
        connector.search(keywords=[], country="TN")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert len(responses.calls) == 0
