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
