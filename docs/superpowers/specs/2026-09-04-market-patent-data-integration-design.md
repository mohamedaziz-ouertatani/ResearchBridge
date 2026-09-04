# Market/Patent Data Integration — Design Spec

Source of truth for the field this replaces: `src/researchbridge/assessment/external_validation.py`,
currently a deterministic function that always returns the same disclaimer text — see that module's
docstring, which cites blueprint Sec 21/48: "There is no external market/patent/industry data source
integrated into this system (Sec 48 is future work)." This spec is that future work, narrowly: it
replaces the static disclaimer with real patent and market-indicator lookups, scoped to Tunisia.

## Why this is a first for the codebase

Every existing `assess_*` function is either fully deterministic over already-ingested/already-
extracted data (novelty, gap, applications, feasibility) or an opt-in local-only LLM call
(opportunity synthesis, gated by `OLLAMA_ENABLED`). Nothing in `build_assessment()` today makes a
live call to an external network service as a hard part of assessment — every existing connector
(`connectors/arxiv.py`, `springer.py`, `semantic_scholar.py`, `core.py`) runs offline, in a batch
ingestion CLI, populating the `papers` table ahead of time; assessment itself only ever reads from
the DB. This spec adds the first query-time external API calls inside the assessment pipeline
itself, so error handling and fail-closed behavior (see below) matter more here than for any
existing stage.

## Scope

In scope:
- A patent-search client against EPO's Open Patent Services (OPS) REST API, scoped to Tunisia
  (`pa=TN` or `pub=TN` in CQL), using a free EPO developer consumer key/secret.
- A market/economic-indicator client against the World Bank Open Data REST API (no auth), pulling a
  fixed small set of Tunisia indicators.
- Deterministic keyword extraction from the `ResearchInput`'s own `title`/`raw_text`, used as the
  query for both clients — no invented terms, matching this codebase's grounding discipline
  elsewhere (novelty/applications/gap all only ever surface text that was actually retrieved).
- Rewriting `assess_external_validation()` to call both clients and format real findings into
  `external_validation_needed` (kept as the existing `Text` column — no schema migration), falling
  back per-source to today's disclaimer wording on any failure.

Out of scope (deliberately deferred):
- Any paid market-intelligence API (Crunchbase, PitchBook, CB Insights) — free/public sources only,
  per the project's existing no-paid-external-API stance (every current connector is free).
- Patents outside Tunisia, or a country-selectable scope — this is the one country the product
  targets right now; broadening is a separate future decision, not implied by this spec.
- Any LLM involvement in this stage. Formatting stays deterministic string templating, same style as
  the current disclaimer text — there's nothing here that needs synthesis, only real lookups
  formatted for reading.
- Persisting patent/market results into their own DB tables, or ingesting patents as corpus items
  searchable like papers. These are per-assessment, point-in-time lookups, not a corpus to retrieve
  against later — closer in spirit to a live API call than to `papers`/`connectors` ingestion.
- Re-running this stage independently after the fact (no "refresh market data" button) — it runs
  once, synchronously, as part of `build_assessment()`, same as every other stage. A user who wants
  fresh data re-runs the whole assessment via the existing rerun flow.

## Components

**`src/researchbridge/assessment/keywords.py`** (new):
```python
def extract_keywords(title: str | None, raw_text: str, max_keywords: int = 8) -> list[str]:
    """Top max_keywords most frequent non-stopword unigrams/bigrams from
    title + raw_text, via sklearn's CountVectorizer(stop_words="english",
    ngram_range=(1, 2)). Deterministic, no training/corpus needed - counts
    are taken over this single document. Returns [] if the combined text
    has no terms surviving stopword removal (e.g. very short/symbol-only
    input) - callers treat that as "nothing to search," not an error."""
```
Pure function, no I/O — testable with plain string fixtures.

**`src/researchbridge/connectors/epo_patents.py`** (new):
```python
@dataclass
class PatentHit:
    title: str
    publication_number: str
    applicant: str | None
    publication_date: date | None
    url: str | None

class EPOPatentConnectorError(Exception):
    """Any failure: auth, network, timeout, non-2xx, unparseable response."""

class EPOPatentConnector:
    def __init__(self, consumer_key: str | None, consumer_secret: str | None, ...): ...
    def search(self, keywords: list[str], country: str = "TN", limit: int = 5) -> list[PatentHit]:
        """Raises EPOPatentConnectorError on any failure. Returns [] (not an
        error) when the search succeeds but finds nothing."""
```
OAuth2 client-credentials token fetch (cached in-memory for the token's stated lifetime, re-fetched
on expiry/401), then a CQL search request (`ti=<keywords> AND pa=TN`, OR-joined keyword terms) — same
`requests` + `tenacity` retry-on-5xx/429 pattern as `springer.py`. Env vars `EPO_OPS_CONSUMER_KEY` /
`EPO_OPS_CONSUMER_SECRET`, added to `.env.example` alongside the existing connector keys. If either
is unset, `EPOPatentConnector.__init__` raises `ValueError` immediately (same pattern as
`SpringerConnector` requiring `api_key`) — the caller in `assess_external_validation()` catches this
alongside `EPOPatentConnectorError` and treats it as "patent source unavailable," not a hard failure.

**`src/researchbridge/connectors/world_bank.py`** (new):
```python
@dataclass
class IndicatorValue:
    indicator_name: str
    value: float
    year: int
    unit: str | None

class WorldBankConnectorError(Exception):
    """Network/timeout/non-2xx/unparseable response."""

class WorldBankConnector:
    def fetch_indicators(self, country: str = "TN") -> list[IndicatorValue]:
        """Fixed indicator set (see INDICATORS below). Raises
        WorldBankConnectorError on failure. An individual indicator with no
        data for the country is silently omitted, not an error - only a
        total request failure raises."""
```
No auth. Fixed `INDICATORS` module constant mapping World Bank indicator codes to display names,
initially: GDP (current US$, `NY.GDP.MKTP.CD`), R&D expenditure (% of GDP, `GB.XPD.RSDV.GD.ZS`),
high-technology exports (% of manufactured exports, `TX.VAL.TECH.MF.ZS`). One request per indicator
(`api.worldbank.org/v2/country/TN/indicator/<code>?format=json&mrnev=1` — most-recent-non-empty-
value), each independently caught so one indicator's absence doesn't drop the others.

**`src/researchbridge/assessment/external_validation.py`** (rewritten):
```python
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
    """patent_connector/market_connector are None when their required env
    vars are unset (World Bank's is always constructible - no auth - but
    accepts None too, for test injection and symmetry). Each source is
    tried independently and caught independently; a source that isn't
    configured, errors, times out, or returns nothing falls back to that
    source's slice of today's disclaimer text. text is always non-empty."""
```
`patent_hits`/`indicator_values` are returned alongside `text` so `build_assessment()` can pass them
to evidence/claims bookkeeping if a later spec wants that (not this one — see Out of scope); for
now `build.py` only uses `.text`.

## Data flow

```
build_assessment()
  keywords = extract_keywords(research_input.title, research_input.raw_text)
  patent_connector = _build_patent_connector()   # None if env vars unset
  market_connector = WorldBankConnector()
  ...
  external_validation = assess_external_validation(
      title=research_input.title,
      raw_text=research_input.raw_text,
      has_applications=bool(applications.applications),
      patent_connector=patent_connector,
      market_connector=market_connector,
  )
  ...
  external_validation_needed = external_validation.text
```

Inside `assess_external_validation`:
```
if keywords is empty:
    return today's full disclaimer (both sources) - matches existing "insufficient
    input to say anything" behavior, nothing to search for
patents:
    if patent_connector is None -> patent section = disclaimer sentence for patents
    else: try patent_connector.search(keywords); on EPOPatentConnectorError or []
          -> disclaimer sentence for patents; on hits -> formatted list
market:
    try market_connector.fetch_indicators(); on WorldBankConnectorError or []
    -> disclaimer sentence for market data; on values -> formatted list
combine both sections + has_applications suffix (unchanged from today) into text
```

## Output formatting

Kept as plain `Text`, deterministic string templating — no schema/JSONB migration, no export.py
change needed (it already treats this field as one text `body`). Example with both sources
succeeding:

```
Market potential, economic impact, and commercialization viability: partial external evidence
found (see below). This does not replace market research or industry expert review.

Related patents (Tunisia, EPO OPS search on "irrigation sensor", "soil moisture"):
- TN12345B1 - "Automated irrigation control system" (applicant: ..., 2023-04-11)
  https://register.epo.org/...
- No other matches found.

Tunisia economic indicators (World Bank, most recent available):
- R&D expenditure: 0.6% of GDP (2021)
- High-technology exports: 3.2% of manufactured exports (2022)
- GDP: $46.3B (2023)

This applies to the potential applications identified above: their real-world demand and
commercial viability still need independent validation.
```

When a source is unavailable/empty, its section reverts to a sentence matching today's tone, e.g.
"Related patents: NOT ASSESSED — patent search unavailable" or "no matching patents found for this
idea's terms" (empty-but-successful search gets a distinct message from a failed/unconfigured one,
so a reader can tell "we looked and found nothing" from "we couldn't look").

## Error handling

- Missing EPO credentials: patent section shows an unconfigured-source message; market section
  proceeds independently (World Bank needs no auth, always attempted).
- EPO auth/network/timeout/5xx/429 (after `tenacity` retries exhaust): caught as
  `EPOPatentConnectorError`, patent section falls back to disclaimer wording; does not affect the
  market section or fail the assessment.
- World Bank network/timeout/5xx: caught as `WorldBankConnectorError`, market section falls back
  similarly; does not affect the patent section.
- Empty `keywords` (degenerate input): both sections skip the network call entirely and fall back
  directly — no point querying with no terms.
- No scenario here raises out of `assess_external_validation()` or `build_assessment()` — this stage
  can never fail an assessment run, matching every existing `assess_*` function's contract.

## Testing

- `keywords.py`: pure-function tests — known input text → expected top terms; empty/whitespace/
  symbol-only input → `[]`; stopwords excluded; bigrams captured (e.g. "soil moisture" as one term,
  not two).
- `epo_patents.py`: `responses`-mocked HTTP — token fetch + search happy path; token expiry triggers
  re-fetch; 401/429/5xx trigger retry then `EPOPatentConnectorError`; empty result set returns `[]`
  (not an error); missing consumer key/secret raises `ValueError` at construction.
- `world_bank.py`: `responses`-mocked HTTP — happy path with all three indicators present; one
  indicator missing data (omitted, others still returned); full request failure raises
  `WorldBankConnectorError`.
- `external_validation.py`: table-driven over the failure matrix — both sources succeed, one
  succeeds/one disclaimer'd (each direction), both disclaimer'd, empty keywords short-circuits
  without calling either connector (assert via mock `assert_not_called`), `has_applications`
  suffix present/absent unchanged from today's behavior.
- `build.py` integration test: one assessment run with both connectors mocked to succeed, confirming
  `external_validation_needed` contains the formatted text and the rest of the pipeline is
  unaffected by this stage's presence.
- No live API calls anywhere in the test suite (matches every existing connector's test approach) —
  `EPO_OPS_CONSUMER_KEY`/`SECRET` unset in CI, so any test needing a "configured" connector injects
  fake credentials directly into the constructor rather than relying on env.

## Migration safety

No DB schema change — `external_validation_needed` stays the existing nullable `Text` column. Fully
backward compatible: an environment with no EPO credentials set gets a patent section that always
reads "unconfigured" while the market section (no auth needed) still works — better than today's
always-empty disclaimer for free, and the system degrades to something close to today's exact text
if `WorldBankConnectorError` is also hit or keywords extraction is empty. No existing test or caller
of `assess_external_validation` keeps its old two-argument signature, since the signature is
changing (adding `title`, `raw_text`, `patent_connector`, `market_connector`) — this is a breaking
change to that one function's call site, confined to `build.py` and that module's own test file.

## Open questions for whoever implements this

1. EPO OPS's CQL syntax for combining multiple extracted keywords (AND vs OR, phrase vs term
   matching) affects precision a lot — worth a short manual exploration against the real sandbox API
   during implementation to see which combination returns usably-relevant results for a few sample
   research ideas, rather than guessing the query shape up front.
2. World Bank's indicator set (GDP, R&D %, high-tech exports) is a reasonable first guess but not
   validated against what a reader actually finds useful for "is this idea commercially viable in
   Tunisia" — worth revisiting once this is live and someone reacts to a real report.
3. `PatentHit`/`IndicatorValue` are returned from `assess_external_validation()` but not persisted
   anywhere beyond the formatted text today (see Out of scope). If a future spec wants them as real
   `Evidence`-backed, citable rows (matching how every other field grounds itself), that's a
   follow-up, not blocking this one — the dataclasses exist now so that follow-up doesn't need this
   module's internals reshaped.
