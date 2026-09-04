from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from researchbridge.fulltext.core_fetch import fetch_core_fulltext


def test_returns_fulltext_when_present(monkeypatch) -> None:
    import researchbridge.fulltext.core_fetch as core_fetch_module

    response = Mock(json=Mock(return_value={"fullText": "the paper's full text"}), raise_for_status=Mock())
    mock_get = Mock(return_value=response)
    monkeypatch.setattr(core_fetch_module.requests, "get", mock_get)

    result = fetch_core_fulltext("123", api_key="test-key")

    assert result == "the paper's full text"
    mock_get.assert_called_once_with(
        "https://api.core.ac.uk/v3/outputs/123", headers={"Authorization": "Bearer test-key"}, timeout=30
    )


def test_returns_none_when_fulltext_is_absent(monkeypatch) -> None:
    import researchbridge.fulltext.core_fetch as core_fetch_module

    response = Mock(json=Mock(return_value={"fullText": None}), raise_for_status=Mock())
    monkeypatch.setattr(core_fetch_module.requests, "get", Mock(return_value=response))

    assert fetch_core_fulltext("123", api_key="test-key") is None


def test_returns_none_when_fulltext_key_is_missing(monkeypatch) -> None:
    import researchbridge.fulltext.core_fetch as core_fetch_module

    response = Mock(json=Mock(return_value={}), raise_for_status=Mock())
    monkeypatch.setattr(core_fetch_module.requests, "get", Mock(return_value=response))

    assert fetch_core_fulltext("123", api_key="test-key") is None


def test_raises_on_http_failure(monkeypatch) -> None:
    import researchbridge.fulltext.core_fetch as core_fetch_module

    monkeypatch.setattr(
        core_fetch_module.requests, "get", Mock(side_effect=requests.ConnectionError("connection refused"))
    )

    with pytest.raises(requests.RequestException):
        fetch_core_fulltext("123", api_key="test-key")
