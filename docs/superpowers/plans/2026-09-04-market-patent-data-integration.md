# Market/Patent Data Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static "market/patent data not integrated" disclaimer in
`external_validation_needed` with real, Tunisia-scoped patent and economic-indicator lookups.

**Architecture:** Two new stateless query-time connectors (EPO OPS for patents, World Bank Open Data
for economic indicators) plus a deterministic keyword extractor, wired into a rewritten
`assess_external_validation()` that formats real findings and falls back per-source to today's
disclaimer wording on any failure. No schema changes; no LLM involvement.

**Tech Stack:** Python 3.13, `requests` + `tenacity` (existing deps), `sklearn.feature_extraction.text`
(existing dep), `responses` for HTTP mocking in tests (existing dev dep), `defusedxml` (new dep) for
parsing EPO OPS's XML responses — stdlib `xml.etree.ElementTree` is XXE/entity-expansion-vulnerable
against untrusted external XML, which this is (a live third-party API response).

**Spec:** [docs/superpowers/specs/2026-09-04-market-patent-data-integration-design.md](../specs/2026-09-04-market-patent-data-integration-design.md)

## Global Constraints

- Free/public data sources only — no paid market-intelligence API.
- Patents/market scope is Tunisia only (`TN`) — no country-selectable parameter.
- No LLM involvement anywhere in this stage — deterministic string formatting only.
- No DB schema migration — `external_validation_needed` stays the existing nullable `Text` column.
- This stage must never raise out of `assess_external_validation()` or fail an assessment run —
  every failure mode falls back to disclaimer-style text instead.
- No live API calls in the test suite — all HTTP mocked via `responses`; missing/fake credentials
  used for connector construction in tests.

---

### Task 1: Deterministic keyword extraction

**Files:**
- Create: `src/researchbridge/assessment/keywords.py`
- Test: `tests/test_assessment_keywords.py`

**Interfaces:**
- Produces: `extract_keywords(title: str | None, raw_text: str, max_keywords: int = 8) -> list[str]`,
  used by Task 4 (`external_validation.py`) and Task 5 (`build.py`).

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from researchbridge.assessment.keywords import extract_keywords


def test_extracts_top_terms_from_title_and_text() -> None:
    keywords = extract_keywords(
        title="Automated irrigation control",
        raw_text="A low-cost soil moisture sensor network for automated irrigation control "
        "in smallholder farms. The system uses soil moisture readings to trigger irrigation.",
        max_keywords=5,
    )

    assert "soil moisture" in keywords
    assert "irrigation" in " ".join(keywords)
    assert len(keywords) <= 5


def test_excludes_stopwords() -> None:
    keywords = extract_keywords(title=None, raw_text="the a an of and or but soil moisture sensor")

    assert "the" not in keywords
    assert "and" not in keywords


def test_empty_text_returns_empty_list() -> None:
    assert extract_keywords(title=None, raw_text="") == []


def test_symbol_only_text_returns_empty_list() -> None:
    assert extract_keywords(title=None, raw_text="!!! ... ---") == []


def test_none_title_does_not_error() -> None:
    keywords = extract_keywords(title=None, raw_text="soil moisture sensor network irrigation control")

    assert keywords != []


def test_is_deterministic() -> None:
    args = {"title": "Irrigation", "raw_text": "soil moisture sensor network for irrigation control"}
    assert extract_keywords(**args) == extract_keywords(**args)


def test_respects_max_keywords() -> None:
    keywords = extract_keywords(
        title="Irrigation",
        raw_text="soil moisture sensor network automated irrigation control valve pump reservoir "
        "farm crop yield water conservation drought",
        max_keywords=3,
    )

    assert len(keywords) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_keywords.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.assessment.keywords'`

- [ ] **Step 3: Write the implementation**

```python
"""Deterministic keyword extraction for external market/patent search queries.

No invented terms - every keyword is text the user actually wrote, matching
this codebase's grounding discipline elsewhere (novelty/applications/gap all
only ever surface text that was actually retrieved). Uses sklearn's
CountVectorizer purely as a frequency counter over the single input document
- no training, no external corpus, no IDF weighting needed since there is
only ever one document per call.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer


def extract_keywords(title: str | None, raw_text: str, max_keywords: int = 8) -> list[str]:
    """Top max_keywords most frequent non-stopword unigrams/bigrams from
    title + raw_text. Returns [] if nothing survives stopword removal (e.g.
    empty, whitespace-only, or symbol-only input) - callers treat that as
    "nothing to search," not an error."""
    combined = f"{title or ''} {raw_text}".strip()
    if not combined:
        return []

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        counts = vectorizer.fit_transform([combined])
    except ValueError:
        # sklearn raises ValueError when every token is a stopword or the
        # vocabulary is otherwise empty (e.g. symbol-only input) - treat
        # the same as "no keywords found", not a crash.
        return []

    terms = vectorizer.get_feature_names_out()
    frequencies = counts.toarray()[0]
    ranked = sorted(zip(terms, frequencies, strict=True), key=lambda pair: (-pair[1], pair[0]))
    return [term for term, _count in ranked[:max_keywords]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_keywords.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/keywords.py tests/test_assessment_keywords.py
git commit -m "feat(assessment): add deterministic keyword extraction for external data queries"
```

---

### Task 2: World Bank economic-indicator connector

**Files:**
- Create: `src/researchbridge/connectors/world_bank.py`
- Create: `tests/fixtures/world_bank_gdp.json`
- Create: `tests/fixtures/world_bank_no_data.json`
- Test: `tests/test_world_bank_connector.py`

**Interfaces:**
- Produces: `IndicatorValue` dataclass (`indicator_name: str`, `value: float`, `year: int`,
  `unit: str | None`), `WorldBankConnectorError` exception, `WorldBankConnector` class with
  `fetch_indicators(self, country: str = "TN") -> list[IndicatorValue]`. Used by Task 4
  (`external_validation.py`) and Task 5 (`build.py`).

- [ ] **Step 1: Create the fixtures**

`tests/fixtures/world_bank_gdp.json`:
```json
[
  {
    "page": 1,
    "pages": 1,
    "per_page": 1,
    "total": 1
  },
  [
    {
      "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
      "country": {"id": "TN", "value": "Tunisia"},
      "countryiso3code": "TUN",
      "date": "2023",
      "value": 46255510000.0,
      "unit": "",
      "obs_status": "",
      "decimal": 0
    }
  ]
]
```

`tests/fixtures/world_bank_no_data.json`:
```json
[
  {
    "page": 1,
    "pages": 1,
    "per_page": 1,
    "total": 0
  },
  null
]
```

- [ ] **Step 2: Write the failing tests**

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_world_bank_connector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.connectors.world_bank'`

- [ ] **Step 4: Write the implementation**

```python
"""World Bank Open Data connector - Tunisia economic/R&D indicators.

No auth required. One request per indicator (mrnev=1 = most-recent-non-
empty-value), each caught independently so one indicator's absence or
failure doesn't drop the others - only a total failure (every indicator
request fails) raises WorldBankConnectorError, matching this module's
"fail closed per source, not per assessment" contract (see
assessment/external_validation.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

WORLD_BANK_API_BASE = "https://api.worldbank.org/v2"

# code -> (display name, unit). Chosen as a first-pass set relevant to
# "is this idea commercially viable" - see spec's open question #2 on
# revisiting this set once real reports are reviewed.
_INDICATORS: dict[str, tuple[str, str]] = {
    "NY.GDP.MKTP.CD": ("GDP", "current US$"),
    "GB.XPD.RSDV.GD.ZS": ("R&D expenditure", "% of GDP"),
    "TX.VAL.TECH.MF.ZS": ("High-technology exports", "% of manufactured exports"),
}


@dataclass
class IndicatorValue:
    indicator_name: str
    value: float
    year: int
    unit: str | None


class WorldBankConnectorError(Exception):
    """Raised only when every configured indicator's request fails -
    network/timeout/non-2xx/unparseable response for all of them. A single
    indicator failing while others succeed does not raise this - see
    fetch_indicators."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True
        return response.status_code == 429 or response.status_code >= 500
    return False


class WorldBankConnector:
    def indicator_codes(self) -> list[str]:
        return list(_INDICATORS.keys())

    def fetch_indicators(self, country: str = "TN") -> list[IndicatorValue]:
        values: list[IndicatorValue] = []
        failures = 0
        for code in self.indicator_codes():
            try:
                value = self._fetch_one(country, code)
            except requests.exceptions.RequestException as exc:
                logger.warning("World Bank indicator %s fetch failed: %s", code, exc)
                failures += 1
                continue
            if value is not None:
                values.append(value)

        if failures == len(self.indicator_codes()):
            raise WorldBankConnectorError(f"all {failures} indicator requests failed for country={country}")
        return values

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_one(self, country: str, code: str) -> IndicatorValue | None:
        response = requests.get(
            f"{WORLD_BANK_API_BASE}/country/{country}/indicator/{code}",
            params={"format": "json", "mrnev": 1},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload[1] if len(payload) > 1 else None
        if not records:
            return None
        record = records[0]
        name, unit = _INDICATORS[code]
        return IndicatorValue(indicator_name=name, value=float(record["value"]), year=int(record["date"]), unit=unit)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_world_bank_connector.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/connectors/world_bank.py tests/test_world_bank_connector.py tests/fixtures/world_bank_gdp.json tests/fixtures/world_bank_no_data.json
git commit -m "feat(connectors): add World Bank economic-indicator connector"
```

---

### Task 3: EPO OPS patent connector

**Files:**
- Create: `src/researchbridge/connectors/epo_patents.py`
- Create: `tests/fixtures/epo_ops_search_results.xml`
- Create: `tests/fixtures/epo_ops_no_results.xml`
- Test: `tests/test_epo_patents_connector.py`

**Interfaces:**
- Produces: `PatentHit` dataclass (`title: str`, `publication_number: str`, `applicant: str | None`,
  `publication_date: date | None`, `url: str | None`), `EPOPatentConnectorError` exception,
  `EPOPatentConnector` class with `__init__(self, consumer_key: str | None, consumer_secret: str | None)`
  (raises `ValueError` if either is falsy) and
  `search(self, keywords: list[str], country: str = "TN", limit: int = 5) -> list[PatentHit]`.
  Used by Task 4 (`external_validation.py`) and Task 5 (`build.py`).

- [ ] **Step 1: Add the `defusedxml` dependency**

```bash
uv add defusedxml
```

This adds `defusedxml` to `pyproject.toml`'s `dependencies` list and updates `uv.lock`. EPO OPS
responses are third-party network input; stdlib `xml.etree.ElementTree` is vulnerable to XXE and
billion-laughs entity-expansion attacks on untrusted XML, so this connector parses with
`defusedxml.ElementTree` instead — same API surface as the stdlib module, just hardened.

- [ ] **Step 2: Create the fixtures**

`tests/fixtures/epo_ops_search_results.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns="http://www.epo.org/exchange">
  <ops:biblio-search total-result-count="1">
    <ops:search-result>
      <ops:publication-reference>
        <document-id document-id-type="epodoc">
          <exchange-document>
            <bibliographic-data>
              <invention-title lang="en">Automated irrigation control system</invention-title>
              <publication-reference>
                <document-id document-id-type="docdb">
                  <country>TN</country>
                  <doc-number>12345</doc-number>
                  <kind>B1</kind>
                  <date>20230411</date>
                </document-id>
              </publication-reference>
              <parties>
                <applicants>
                  <applicant>
                    <applicant-name>
                      <name>Example Agritech SARL</name>
                    </applicant-name>
                  </applicant>
                </applicants>
              </parties>
            </bibliographic-data>
          </exchange-document>
        </document-id>
      </ops:publication-reference>
    </ops:search-result>
  </ops:biblio-search>
</ops:world-patent-data>
```

`tests/fixtures/epo_ops_no_results.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns="http://www.epo.org/exchange">
  <ops:biblio-search total-result-count="0">
  </ops:biblio-search>
</ops:world-patent-data>
```

- [ ] **Step 3: Write the failing tests**

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_epo_patents_connector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.connectors.epo_patents'`

- [ ] **Step 5: Write the implementation**

```python
"""EPO Open Patent Services (OPS) connector - Tunisia patent search.

OAuth2 client-credentials flow (POST to EPO_OPS_TOKEN_URL with HTTP Basic
auth, grant_type=client_credentials), then a CQL bibliographic search
(GET EPO_OPS_SEARCH_URL?q=...). The token is cached in-memory for its
stated expires_in lifetime and re-fetched on expiry or a 401 from the
search endpoint.

Query syntax note (see spec's open question #1 - this needs a live sandbox
check during rollout): keywords are OR-joined into a title/abstract search,
AND-ed with a country filter for publication authority - e.g. for
keywords=["irrigation", "soil moisture"], country="TN":
    (ti="irrigation" OR ti="soil moisture") AND pn=TN

Response is XML (ops:world-patent-data namespace) - parsed with
defusedxml.ElementTree, not the stdlib xml.etree.ElementTree: this is
third-party network input (a live external API response), and stdlib XML
parsing is vulnerable to XXE/entity-expansion attacks on untrusted input.
defusedxml.ElementTree is a drop-in replacement with the same API.
"""

from __future__ import annotations

import time
from base64 import b64encode
from dataclasses import dataclass
from datetime import date, datetime
from xml.etree.ElementTree import ParseError

import requests
from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

EPO_OPS_TOKEN_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_OPS_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"

_NS = {"ops": "http://ops.epo.org", "ex": "http://www.epo.org/exchange"}


@dataclass
class PatentHit:
    title: str
    publication_number: str
    applicant: str | None
    publication_date: date | None
    url: str | None


class EPOPatentConnectorError(Exception):
    """Any failure: auth, network, timeout, non-2xx, unparseable response."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
        if response is None:
            return True
        return response.status_code == 429 or response.status_code >= 500
    return False


class EPOPatentConnector:
    def __init__(self, consumer_key: str | None, consumer_secret: str | None) -> None:
        if not consumer_key or not consumer_secret:
            raise ValueError("consumer_key and consumer_secret are required (see EPO_OPS_CONSUMER_KEY/SECRET in .env.example)")
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def search(self, keywords: list[str], country: str = "TN", limit: int = 5) -> list[PatentHit]:
        if not keywords:
            raise ValueError("keywords must not be empty")
        try:
            token = self._get_token()
            query = self._build_query(keywords, country)
            response = self._fetch_search(token, query, limit)
            return self._parse_hits(response.text)[:limit]
        except (requests.exceptions.RequestException, ParseError, DefusedXmlException) as exc:
            raise EPOPatentConnectorError(f"EPO OPS search failed: {exc}") from exc

    def _build_query(self, keywords: list[str], country: str) -> str:
        title_clause = " OR ".join(f'ti="{kw}"' for kw in keywords)
        return f"({title_clause}) AND pn={country}"

    def _get_token(self) -> str:
        if self._token is not None and time.monotonic() < self._token_expires_at:
            return self._token
        credentials = b64encode(f"{self.consumer_key}:{self.consumer_secret}".encode()).decode()
        response = requests.post(
            EPO_OPS_TOKEN_URL,
            headers={"Authorization": f"Basic {credentials}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.monotonic() + float(payload.get("expires_in", 1200)) - 30
        return self._token

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_search(self, token: str, query: str, limit: int) -> requests.Response:
        response = requests.get(
            EPO_OPS_SEARCH_URL,
            params={"q": query, "Range": f"1-{limit}"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        if response.status_code == 401:
            # token may have been invalidated server-side before our cached
            # expiry - force one re-fetch and retry once, not via tenacity
            # (that retries the same request, not a fresh token).
            self._token = None
            token = self._get_token()
            response = requests.get(
                EPO_OPS_SEARCH_URL,
                params={"q": query, "Range": f"1-{limit}"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
        response.raise_for_status()
        return response

    def _parse_hits(self, xml_text: str) -> list[PatentHit]:
        root = ET.fromstring(xml_text)
        hits: list[PatentHit] = []
        for doc in root.iterfind(".//ex:exchange-document", _NS):
            biblio = doc.find("ex:bibliographic-data", _NS)
            if biblio is None:
                continue
            title_el = biblio.find("ex:invention-title", _NS)
            title = title_el.text if title_el is not None and title_el.text else "Untitled"

            pub_ref = biblio.find("ex:publication-reference/ex:document-id", _NS)
            country_el = pub_ref.find("ex:country", _NS) if pub_ref is not None else None
            number_el = pub_ref.find("ex:doc-number", _NS) if pub_ref is not None else None
            kind_el = pub_ref.find("ex:kind", _NS) if pub_ref is not None else None
            date_el = pub_ref.find("ex:date", _NS) if pub_ref is not None else None

            country_text = country_el.text if country_el is not None else ""
            number_text = number_el.text if number_el is not None else ""
            kind_text = kind_el.text if kind_el is not None else ""
            publication_number = f"{country_text}{number_text}{kind_text}"

            publication_date = _parse_date(date_el.text if date_el is not None else None)

            applicant_el = biblio.find("ex:parties/ex:applicants/ex:applicant/ex:applicant-name/ex:name", _NS)
            applicant = applicant_el.text if applicant_el is not None else None

            hits.append(
                PatentHit(
                    title=title,
                    publication_number=publication_number,
                    applicant=applicant,
                    publication_date=publication_date,
                    url=f"https://register.epo.org/application?number={publication_number}" if publication_number else None,
                )
            )
        return hits


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_epo_patents_connector.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/researchbridge/connectors/epo_patents.py tests/test_epo_patents_connector.py tests/fixtures/epo_ops_search_results.xml tests/fixtures/epo_ops_no_results.xml
git commit -m "feat(connectors): add EPO OPS Tunisia patent-search connector"
```

---

### Task 4: Rewrite `assess_external_validation()` to use real data

**Files:**
- Modify: `src/researchbridge/assessment/external_validation.py` (full rewrite, currently 42 lines)
- Modify: `tests/test_assessment_external_validation.py` (full rewrite, currently 35 lines)

**Interfaces:**
- Consumes: `extract_keywords` (Task 1), `EPOPatentConnector`/`PatentHit`/`EPOPatentConnectorError`
  (Task 3), `WorldBankConnector`/`IndicatorValue`/`WorldBankConnectorError` (Task 2).
- Produces: `ExternalValidationResult` dataclass (`text: str`, `patent_hits: list[PatentHit]`,
  `indicator_values: list[IndicatorValue]`), and
  `assess_external_validation(title: str | None, raw_text: str, has_applications: bool, patent_connector: EPOPatentConnector | None, market_connector: WorldBankConnector | None) -> ExternalValidationResult`.
  Used by Task 5 (`build.py`).

- [ ] **Step 1: Write the failing tests (replaces the existing file entirely)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assessment_external_validation.py -v`
Expected: FAIL (old `assess_external_validation(has_applications=...)` signature doesn't match new
calls; `TypeError: assess_external_validation() got an unexpected keyword argument 'title'`)

- [ ] **Step 3: Write the implementation**

```python
"""External validation (blueprint Sec 21, Layer 3 - External Impact; Sec 48).

Real patent and market-indicator lookups, scoped to Tunisia, replacing the
previous always-identical disclaimer (see git history for that version).
Sec 21 still governs the framing: market potential, economic impact, and
commercialization viability are never confidently ASSERTED from literature
or from a handful of patent/indicator hits - this stage only ever reports
what real external sources returned, with an explicit "still needs
independent validation" framing, never a market-potential LEVEL judgment.

Fail-closed per source (see docs/superpowers/specs/2026-09-04-market-patent
-data-integration-design.md): an unconfigured, erroring, timing-out, or
empty-keyword-blocked source reverts to disclaimer wording for its own
section only - this function never raises, and the overall assessment
pipeline never fails because of this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchbridge.assessment.keywords import extract_keywords
from researchbridge.connectors.epo_patents import EPOPatentConnector, EPOPatentConnectorError, PatentHit
from researchbridge.connectors.world_bank import IndicatorValue, WorldBankConnector, WorldBankConnectorError

_DISCLAIMER_TEXT = (
    "Market potential, economic impact, and commercialization viability are "
    "NOT ASSESSED. This system does not integrate external market, patent, "
    "or industry data (see blueprint Sec 21/48) - the literature alone cannot "
    "support a judgment here. Required validation: market research and/or "
    "industry expert review."
)

_APPLICATIONS_SUFFIX = (
    " This applies to the potential applications identified above: their real-world "
    "demand and commercial viability still need independent validation."
)

_INTRO_TEXT = (
    "Market potential, economic impact, and commercialization viability: external evidence "
    "found below (see sources). This does not replace independent market research or industry "
    "expert review."
)

_PATENT_UNCONFIGURED = "Related patents: NOT ASSESSED - patent search unavailable."
_PATENT_ERROR = "Related patents: NOT ASSESSED - patent search failed."
_PATENT_NONE_FOUND = "Related patents (Tunisia): no matching patents found for this idea's terms."
_MARKET_ERROR = "Tunisia economic indicators: NOT ASSESSED - indicator lookup failed."
_MARKET_NONE_FOUND = "Tunisia economic indicators: no data available."


@dataclass
class ExternalValidationResult:
    text: str
    patent_hits: list[PatentHit]
    indicator_values: list[IndicatorValue]


def assess_external_validation(
    title: str | None,
    raw_text: str,
    has_applications: bool,
    patent_connector: EPOPatentConnector | None,
    market_connector: WorldBankConnector | None,
) -> ExternalValidationResult:
    keywords = extract_keywords(title, raw_text)
    if not keywords:
        return ExternalValidationResult(
            text=_DISCLAIMER_TEXT + (_APPLICATIONS_SUFFIX if has_applications else ""),
            patent_hits=[],
            indicator_values=[],
        )

    patent_hits, patent_section = _patent_section(patent_connector, keywords)
    indicator_values, market_section = _market_section(market_connector)

    text = _INTRO_TEXT + "\n\n" + patent_section + "\n\n" + market_section
    if has_applications:
        text += _APPLICATIONS_SUFFIX

    return ExternalValidationResult(text=text, patent_hits=patent_hits, indicator_values=indicator_values)


def _patent_section(connector: EPOPatentConnector | None, keywords: list[str]) -> tuple[list[PatentHit], str]:
    if connector is None:
        return [], _PATENT_UNCONFIGURED
    try:
        hits = connector.search(keywords=keywords, country="TN")
    except EPOPatentConnectorError:
        return [], _PATENT_ERROR
    if not hits:
        return [], _PATENT_NONE_FOUND
    keyword_list = ", ".join(f'"{kw}"' for kw in keywords)
    lines = [f"Related patents (Tunisia, EPO OPS search on {keyword_list}):"]
    for hit in hits:
        applicant = f" (applicant: {hit.applicant})" if hit.applicant else ""
        pub_date = f" - {hit.publication_date.isoformat()}" if hit.publication_date else ""
        lines.append(f"- {hit.publication_number} - \"{hit.title}\"{applicant}{pub_date}")
        if hit.url:
            lines.append(f"  {hit.url}")
    return hits, "\n".join(lines)


def _market_section(connector: WorldBankConnector | None) -> tuple[list[IndicatorValue], str]:
    if connector is None:
        return [], _MARKET_ERROR
    try:
        values = connector.fetch_indicators(country="TN")
    except WorldBankConnectorError:
        return [], _MARKET_ERROR
    if not values:
        return [], _MARKET_NONE_FOUND
    lines = ["Tunisia economic indicators (World Bank, most recent available):"]
    for value in values:
        unit = f" {value.unit}" if value.unit else ""
        lines.append(f"- {value.indicator_name}: {value.value:,.1f}{unit} ({value.year})")
    return values, "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assessment_external_validation.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/assessment/external_validation.py tests/test_assessment_external_validation.py
git commit -m "feat(assessment): populate external_validation with real patent/market findings"
```

---

### Task 5: Wire connectors into `build_assessment()` and document env vars

**Files:**
- Modify: `src/researchbridge/assessment/build.py:63` (import), `:172` (call site)
- Modify: `.env.example`
- Test: `tests/test_assessment_build.py` (add one integration test to the existing file — real
  fixtures confirmed below, no live network calls needed since `session_factory`/`_research_input`
  are pure-DB, and both connectors are mocked)

**Interfaces:**
- Consumes: `extract_keywords` (Task 1), `EPOPatentConnector` (Task 3), `WorldBankConnector`
  (Task 2), `assess_external_validation` (Task 4).

Note on the plan's earlier task list (Task 5's file was originally drafted from a guessed path/
fixture set before this task ran — corrected here against the real file): the actual test file is
`tests/test_assessment_build.py`, not `tests/test_build_assessment.py`. It has no `session`/
`research_input` fixtures — instead a session-scoped `embedder` fixture (a `FakeEmbedder`
dataclass), a `session_factory` fixture (call it to get a session, `session.close()` it when done —
per this codebase's DB-test convention, an unclosed session can deadlock later TRUNCATE-based test
teardown), and a local `_research_input(session, raw_text) -> ResearchInput` helper already defined
in the file (id/input_type/raw_text only — `title` stays `None`, which is fine: Task 4's
`assess_external_validation` already handles `title=None`).

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_assessment_build.py`:

```python
from unittest.mock import patch

from researchbridge.connectors.epo_patents import PatentHit
from researchbridge.connectors.world_bank import IndicatorValue


def test_external_validation_uses_real_connector_results(session_factory, embedder) -> None:
    session = session_factory()
    ri = _research_input(session, "soil moisture sensor network for automated irrigation control")
    session.commit()

    fake_patent_hit = PatentHit(
        title="Automated irrigation control system",
        publication_number="TN12345B1",
        applicant="Example Agritech SARL",
        publication_date=None,
        url=None,
    )
    fake_indicator = IndicatorValue(indicator_name="GDP", value=1.0, year=2023, unit="current US$")

    with (
        patch("researchbridge.assessment.build._build_patent_connector") as mock_patent_factory,
        patch("researchbridge.assessment.build.WorldBankConnector") as mock_market_cls,
    ):
        mock_patent_factory.return_value.search.return_value = [fake_patent_hit]
        mock_market_cls.return_value.fetch_indicators.return_value = [fake_indicator]

        assessment = build_assessment(session, ri.id, embedder)

    session.close()
    assert "TN12345B1" in assessment.external_validation_needed
    assert "GDP" in assessment.external_validation_needed
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_assessment_build.py::test_external_validation_uses_real_connector_results -v`
Expected: FAIL — `assess_external_validation()` call in `build.py` still uses the old
`has_applications`-only signature, and `_build_patent_connector`/`WorldBankConnector` aren't
imported into `build.py` yet, so the `patch()` targets don't exist
(`AttributeError: <module 'researchbridge.assessment.build'> does not have the attribute
'_build_patent_connector'`)

- [ ] **Step 3: Update `build.py`**

Modify the import block (around line 63):
```python
from researchbridge.assessment.external_validation import assess_external_validation
```
becomes:
```python
import os

from researchbridge.assessment.external_validation import assess_external_validation
from researchbridge.connectors.epo_patents import EPOPatentConnector
from researchbridge.connectors.world_bank import WorldBankConnector


def _build_patent_connector() -> EPOPatentConnector | None:
    consumer_key = os.environ.get("EPO_OPS_CONSUMER_KEY")
    consumer_secret = os.environ.get("EPO_OPS_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        return None
    return EPOPatentConnector(consumer_key=consumer_key, consumer_secret=consumer_secret)
```

(Place the `import os` at the top of the file's existing stdlib import group, and
`_build_patent_connector` as a module-level function near the top, after the imports — matching
this file's existing style of small module-level helpers below the docstring/imports.)

Modify the call site (around line 172):
```python
    external_validation_needed = assess_external_validation(has_applications=bool(applications.applications))
```
becomes:
```python
    external_validation = assess_external_validation(
        title=research_input.title,
        raw_text=research_input.raw_text,
        has_applications=bool(applications.applications),
        patent_connector=_build_patent_connector(),
        market_connector=WorldBankConnector(),
    )
    external_validation_needed = external_validation.text
```

- [ ] **Step 4: Add env vars to `.env.example`**

Append after the `CORE_API_KEY` block:
```
# Optional - powers real patent-search results in the "external validation"
# assessment section (Tunisia-scoped). Register a free consumer key/secret
# at https://developers.epo.org. If unset, that section falls back to its
# "patent search unavailable" disclaimer wording - the rest of the
# assessment is unaffected.
EPO_OPS_CONSUMER_KEY=
EPO_OPS_CONSUMER_SECRET=
```

(World Bank needs no env var — it's unauthenticated, so `WorldBankConnector()` is always
constructed directly in `build.py`, no factory/None-check needed.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_assessment_build.py::test_external_validation_uses_real_connector_results -v`
Expected: PASS

- [ ] **Step 6: Run the full set of touched test files**

Run: `pytest tests/test_assessment_keywords.py tests/test_world_bank_connector.py tests/test_epo_patents_connector.py tests/test_assessment_external_validation.py tests/test_assessment_build.py -v`
Expected: PASS (all tests across all five files)

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/assessment/build.py .env.example tests/test_assessment_build.py
git commit -m "feat(assessment): wire EPO/World Bank connectors into build_assessment()"
```

---

## Post-plan verification

`build.py` is shared pipeline code, so after Task 5 lands, run the broader assessment test
directory (not necessarily the full ~15-17min suite) to catch any test elsewhere that constructed
`ResearchAssessment` rows and asserted on the literal old disclaimer text:

```bash
grep -rl "external_validation_needed\|assess_external_validation\|does not integrate external" tests/ | xargs pytest -v
```

Fix any test still asserting the old static disclaimer wording — update it to assert against the
new fallback-path wording (`_DISCLAIMER_TEXT`/`_PATENT_UNCONFIGURED`/`_MARKET_ERROR` text) instead,
since in CI (no `EPO_OPS_CONSUMER_KEY` set, but `WorldBankConnector` making a real unmocked call
would be a live network call in a test — check whether that grep turns up any test calling
`build_assessment()` without mocking `WorldBankConnector`, and mock it there too, following the
Task 5 pattern).
