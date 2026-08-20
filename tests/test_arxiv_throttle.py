from __future__ import annotations

from pathlib import Path

import responses

from researchbridge.connectors.arxiv import ARXIV_API_URL, MIN_REQUEST_INTERVAL_SECONDS, ArxivConnector

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClock:
    """Manual clock + sleep recorder, so throttling tests never really wait."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _connector(clock: FakeClock, **kwargs) -> ArxivConnector:
    return ArxivConnector(
        search_query="cat:cs.LG",
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        **kwargs,
    )


def _stub_response() -> None:
    responses.add(responses.GET, ARXIV_API_URL, body=(FIXTURES / "arxiv_page1.xml").read_bytes(), status=200)


@responses.activate
def test_first_request_does_not_sleep() -> None:
    _stub_response()
    clock = FakeClock()

    _connector(clock).fetch(resume_state=None)

    assert clock.sleeps == []


@responses.activate
def test_second_immediate_request_waits_the_full_interval() -> None:
    _stub_response()
    _stub_response()
    clock = FakeClock()
    connector = _connector(clock)

    connector.fetch(resume_state=None)
    connector.fetch(resume_state={"start_index": 2})

    assert clock.sleeps == [MIN_REQUEST_INTERVAL_SECONDS]


@responses.activate
def test_elapsed_time_is_deducted_from_the_wait() -> None:
    _stub_response()
    _stub_response()
    clock = FakeClock()
    connector = _connector(clock)

    connector.fetch(resume_state=None)
    clock.advance(1.0)  # a slow response already burned part of the interval
    connector.fetch(resume_state={"start_index": 2})

    assert clock.sleeps == [MIN_REQUEST_INTERVAL_SECONDS - 1.0]


@responses.activate
def test_no_sleep_when_interval_already_elapsed() -> None:
    _stub_response()
    _stub_response()
    clock = FakeClock()
    connector = _connector(clock)

    connector.fetch(resume_state=None)
    clock.advance(MIN_REQUEST_INTERVAL_SECONDS + 0.5)
    connector.fetch(resume_state={"start_index": 2})

    assert clock.sleeps == []


@responses.activate
def test_interval_is_configurable() -> None:
    _stub_response()
    _stub_response()
    clock = FakeClock()
    connector = _connector(clock, min_request_interval=0.5)

    connector.fetch(resume_state=None)
    connector.fetch(resume_state={"start_index": 2})

    assert clock.sleeps == [0.5]
