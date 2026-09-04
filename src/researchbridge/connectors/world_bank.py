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
