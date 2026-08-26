# Local-LLM Summarization Layer for Corpus Q&A — Design Spec

Source of truth for the feature it extends: `docs/superpowers/specs/2026-08-26-corpus-qa-design.md`
(the extractive `/api/ask` design) and `src/researchbridge/qa/answer.py`. That design deliberately
excluded any generative LLM call — "no new dependency, no API key, no free-text synthesis" — as a
load-bearing decision. This spec revisits that decision narrowly: it adds an **optional**,
**off-by-default**, **additive** layer that never replaces or alters the existing extractive
behavior. `POST /api/ask` is unchanged, byte-for-byte, by this feature.

## Scope

In scope:
- A new endpoint, `POST /api/ask/summarize`, that takes a question plus the exact quote hits the
  client already received from `/api/ask`, and returns a short synthesized summary that only
  rephrases/connects those quotes, with inline citations.
- A local Ollama call (`llama3.1:8b` default, configurable) over HTTP to `localhost:11434` via
  `httpx`, matching the existing connector pattern (`connectors/springer.py`,
  `connectors/semantic_scholar.py`) — no new Python dependency.
- Deterministic citation-existence validation: every `[n]` marker in the model's output must
  reference an actual index in the hits the backend was given. One retry on failure, then fail
  closed with no summary shown.
- A deployment-wide config flag (`OLLAMA_ENABLED`, default `false`) gating the entire feature — off
  by default, so existing deployments are unaffected until explicitly turned on.
- A per-request UI toggle: a "synthesize a summary" button on `/ask` that only appears when the
  backend reports the feature is available, and only fires the Ollama call when clicked.
- Raw quotes remain visible, unchanged, and unreplaced at all times — the summary is an additional
  panel above them, not a replacement.

Out of scope (deliberately deferred):
- Any sentence-level entailment/lexical-overlap validation of the summary's wording against its
  cited quote — validation is limited to "does this citation number exist," not "is this sentence
  actually supported by that quote's text." The raw quotes being visible is the mitigation for this
  gap, not an automated check.
- Persisting summaries anywhere (no new table) — stateless like `/api/ask` itself.
- Multi-turn conversation, follow-up questions, or any memory of prior summaries.
- Any model other than Ollama-served local models. No cloud LLM API, no API key.
- Streaming the summary token-by-token — the endpoint returns the complete summary in one response.
- Automatically re-deriving hits server-side; the client must pass back the exact `QuoteHitOut[]`
  it received from `/api/ask`, so there is no drift between what the user sees and what gets cited.

## Data flow

```
frontend already has hits[] from a completed POST /api/ask call
  -> user clicks "synthesize a summary" (only shown if summarization_available)
  -> POST /api/ask/summarize {question, hits}
       -> 503 immediately if OLLAMA_ENABLED is false
       -> build numbered prompt: [1] "quote text" - Paper Title, [2] ..., for each hit in order
       -> httpx POST to OLLAMA_HOST/api/chat, system+user prompt (see Prompt design), temperature ~0.2
       -> parse response text, regex-extract all [n] citation markers
       -> validate: every n is in range 1..len(hits)
            -> valid: return {summary, citations: [n, ...]}
            -> invalid: retry once (same prompt, fresh call)
                 -> still invalid or Ollama unreachable/timeout: 503, no summary
  -> frontend renders summary panel above the quote list, [n] markers become anchor
     links that scroll to and briefly highlight the matching QuoteCard
```

No new table, no migration, no new embedding computation. The only new computation is one (or two,
on retry) HTTP calls to a local Ollama server, made only when a user explicitly requests it.

## Backend

**New module** `src/researchbridge/qa/summarize.py`:

```python
@dataclass
class SummaryResult:
    summary: str
    citations: list[int]  # 1-indexed positions into the hits list that were actually cited

class SummarizationUnavailable(Exception):
    """Raised when OLLAMA_ENABLED is false, Ollama is unreachable, or validation
    fails after retry. Route layer turns this into a 503."""

def build_prompt(question: str, hits: list[QuoteHitOut]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). Pure function, easy to unit test
    without a real Ollama."""
    ...

def extract_citations(text: str, hit_count: int) -> list[int]:
    """Regex-extracts [n] markers, raises ValueError if any n is out of range
    1..hit_count. Pure function."""
    ...

def summarize_quotes(
    question: str, hits: list[QuoteHitOut], settings: Settings,
) -> SummaryResult:
    """Orchestrates: build_prompt -> call Ollama -> extract_citations,
    with one retry on ValueError or Ollama-call failure. Raises
    SummarizationUnavailable if OLLAMA_ENABLED is false or both attempts fail."""
    ...
```

**Prompt design** (system prompt, sent alongside a user prompt containing the numbered quote list
and the question):

> "You are given a question and a numbered list of verbatim quotes from research papers. Write a
> 3–5 sentence synthesis using ONLY information stated in these quotes — never add outside
> knowledge or infer anything not explicitly present. After every sentence, cite the quote
> number(s) it draws from in brackets, e.g. [1] or [1][3]. If the quotes don't address the
> question, say that plainly instead of guessing. Do not use any information not in the numbered
> quotes above."

Temperature ~0.2. The constraint is stated at both the start and end of the system prompt
(repetition measurably reduces drift in small instruct models).

**New config** in the existing settings module: `OLLAMA_ENABLED` (bool, default `false`),
`OLLAMA_MODEL` (str, default `"llama3.1:8b"`), `OLLAMA_HOST` (str, default
`"http://localhost:11434"`), `OLLAMA_TIMEOUT_SECONDS` (int, default `30`).

**New schemas** in `api/schemas.py`:
```python
class SummarizeRequest(BaseModel):
    question: str
    hits: list[QuoteHitOut]

class SummarizeResponse(BaseModel):
    summary: str
    citations: list[int]
```

Also extend `AskResponse` with one new field: `summarization_available: bool` (mirrors
`OLLAMA_ENABLED` from settings) so the frontend knows whether to show the button, without a
separate config round-trip.

**Route** (added to `src/researchbridge/api/qa_routes.py`):
```
POST /api/ask/summarize
  body: {"question": str, "hits": QuoteHitOut[]}
  503 if OLLAMA_ENABLED is false, or Ollama is unreachable/times out, or validation
      fails after one retry — body: {"detail": "..."} explaining which
  200: SummarizeResponse
```

## Frontend

**`frontend/lib/qaApi.ts`**: add `summarize: (question: string, hits: QuoteHit[]) =>
Promise<{summary: string; citations: number[]}>`, and add `summarization_available: boolean` to
the existing `QuoteHit`-adjacent response type for `ask()`.

**`frontend/app/ask/page.tsx`**:
- After `hits` render (and only if `summarization_available`), show a button: "✨ synthesize a
  summary from these quotes".
- Clicking it calls `qaApi.summarize(question, hits)`, shows a busy state in a new panel that
  appears **above** the quote list (per the agreed placement), summary text below.
- Panel is labeled "AI-synthesized from the quotes below — not independently verified" so it never
  reads as authoritative on its own.
- `[n]` markers in the rendered summary become anchor links; each `QuoteCard` gets `id={`quote-
  ${n}`}` and a brief highlight animation on jump-to, reusing the existing `resolve` animation
  pattern already used for card entrance.
- On 503 / network failure: inline error in the panel ("local LLM unavailable — quotes above are
  unaffected"), quotes list is never touched or hidden.
- Quotes render exactly as they do today regardless of summarization state — this feature only
  ever adds a panel above them.

## Error handling

- `OLLAMA_ENABLED=false` (default): `/api/ask` reports `summarization_available: false`, the
  button never renders, `/api/ask/summarize` would 503 if called directly.
- Ollama unreachable, times out, or returns an empty/unparseable response: after one retry,
  `/api/ask/summarize` returns 503 with an explanatory message. Frontend shows inline error,
  quotes unaffected.
- Model cites an out-of-range quote number: treated identically to an unreachable Ollama — retry
  once, then 503. Never silently strip the bad citation and show a partially-validated summary.
- Empty `hits` passed to `/api/ask/summarize` (shouldn't happen from the UI, since the button only
  shows when hits exist, but the route validates defensively): 422.

## Testing

- Backend: unit tests for `build_prompt` (numbering matches hit order) and `extract_citations`
  (valid range, out-of-range raises, no citations found, malformed brackets) — pure functions, no
  Ollama needed. `summarize_quotes` tested with the httpx call mocked: happy path, retry-then-
  succeed, retry-then-fail, timeout. Route tests (`test_qa_api.py`): 200 with valid summary
  (mocked Ollama), 503 when `OLLAMA_ENABLED=false`, 503 when Ollama mocked unreachable, 422 on
  empty hits.
- Frontend: no test infra in this repo (consistent with the rest of the session) — verified live
  via browser-preview against a real local Ollama instance: button only appears when enabled,
  summary renders with working citation links that scroll to and highlight the right quote card,
  disabled/unreachable states render their inline messages correctly, quote list is untouched
  throughout.

## Migration safety

No schema change, no migration. `POST /api/ask` gains one new response field
(`summarization_available`) but all existing behavior and all existing fields are unchanged. The
new endpoint and new frontend button are fully additive and inert unless `OLLAMA_ENABLED=true` is
explicitly set.
