# Potential Product/Technology Opportunities — Design Spec

Source of truth for the field this extends: `src/researchbridge/assessment/opportunities.py`
(currently always returns `OpportunitiesResult(opportunities=None, evidence_ids=[])` — see that
module's docstring for why it was deliberately left unbuilt) and blueprint Sec 33 / the Sec 49
worked example, which requires exactly three tiers:

```text
Direct: Real-time fraud-scoring API for a payment processor
Adjacent: Streaming risk-monitoring platform for multiple transaction types
Speculative: Cross-bank real-time fraud intelligence network
```

This spec revisits the "left unbuilt" decision narrowly, the same way
`docs/superpowers/specs/2026-08-26-ollama-summary-layer-design.md` narrowly revisited corpus
Q&A's "no generative LLM" decision — additive, optional, off by default, never changing existing
behavior when not opted in.

## Why this isn't a normal bugfix

Every other `assess_*` function in this package is extractive/deterministic: RAKE keyword pulls,
cosine similarity, regex validation over already-extracted claim text. `potential_opportunities`
is different in kind — Direct/Adjacent/Speculative product framing requires *inventing* a product
concept that is not literally present in any paper. `opportunities.py`'s docstring already spells
out why nothing in the deterministic toolbox can do this honestly:

> (a) trivially relabeling an existing application as "Direct" — real, but not actually a distinct
> opportunity... or (b) inventing content via a generative model.

This codebase has exactly one precedent for (b) done responsibly: `src/researchbridge/qa/summarize.py`,
merged and live behind `OLLAMA_ENABLED` (default `false`). Same shape of problem — synthesize
free-text from grounded input, using a local model, with citations checked against real IDs before
anything is shown. This spec reuses that infrastructure rather than inventing a second pattern.

## Scope

In scope:
- A synthesis step that takes the assessment's own `potential_applications` (already grounded,
  already evidence-linked — `assess_applications`'s output, never raw retrieval) and asks a local
  Ollama model to propose one Direct, one Adjacent, and one Speculative opportunity, each citing
  which source application(s) it drew from by index.
- Reuse of the existing `OLLAMA_ENABLED` / `OLLAMA_MODEL` / `OLLAMA_HOST` / `OLLAMA_TIMEOUT_SECONDS`
  config — no new flag. Wherever an operator has already turned on the Q&A summary layer, this
  becomes available too; wherever they haven't, `potential_opportunities` stays exactly the honest
  `NULL` it is today. No new deployment decision to make.
- Deterministic citation-existence validation identical in spirit to `extract_citations`: every
  cited application index must be real. One retry, then fail closed to `NULL` — never a
  partially-validated result persisted to the database.
- Triggered on demand, not during `build_assessment()` — see "Where this runs" below.

Out of scope (deliberately deferred, matching the Q&A layer's own exclusions):
- Sentence-level entailment checking of the synthesized text against its cited application's
  wording — validation is "does this citation index exist," not "is this specific claim about the
  product actually supported." The visible source-application link is the mitigation, not an
  automated check (identical stance to the Q&A summary's raw-quotes-stay-visible mitigation).
- Any cloud LLM / API key. Local Ollama only, matching the rest of the codebase's zero-external-
  LLM-dependency stance outside this one exception.
- Re-synthesizing automatically when an assessment is re-run — a rerun creates a new assessment row
  via the existing rerun flow; opportunities synthesis is requested again the same way, on demand.
- Any attempt to synthesize opportunities when `potential_applications` is empty or `null` (no
  relevant applications were ever found) — nothing to ground a product idea in, so this returns
  "not enough evidence" immediately, without calling Ollama at all.

## Where this runs: on-demand, not inside `build_assessment()`

Every other `assess_*` function runs synchronously inside `build_assessment()`, and that pipeline
is currently 100% deterministic and has no external-service dependency — it never fails because
some other process isn't running. Wiring an Ollama call into that path would change that
invariant: assessment creation would either (a) silently degrade (acceptable, but inconsistent
with every other field's all-or-nothing persistence within one `build_assessment()` call) or (b)
become slower and occasionally flaky whenever Ollama is enabled but briefly unresponsive, for
every assessment, even for users who never look at the opportunities section.

The Q&A summary layer already solved this exact tension by making itself a separate, explicitly
user-triggered call (`POST /api/ask/summarize`) rather than folding into `POST /api/ask`. This spec
follows the same shape: a new endpoint, called only when a user asks for it.

## Data flow

```
Assessment report page already has a completed ResearchAssessment with
potential_applications: ApplicationRecord[] (paper_id, application text, source_paper)
  -> if potential_applications is empty/null: no button shown, section stays "not generated"
  -> user clicks "synthesize opportunities" under the opportunities section
  -> POST /api/assessments/{id}/opportunities
       -> 503 immediately if OLLAMA_ENABLED is false
       -> build numbered prompt: [1] "application text" - Paper Title, [2] ..., one per
          potential_applications entry, in existing order
       -> requests.post to OLLAMA_HOST/api/chat (same call shape as summarize.py),
          system+user prompt (see Prompt design), temperature ~0.2
       -> parse response: expect three labeled lines (Direct/Adjacent/Speculative), each
          ending in a [n] citation
       -> validate: exactly three tiers present, each with >=1 valid citation index
            -> valid: persist to potential_opportunities, return 200
            -> invalid (missing tier, malformed line, out-of-range citation, unreachable):
               retry once (same prompt, fresh call) -> still invalid: 503, potential_opportunities
               stays NULL (unchanged)
  -> frontend re-fetches the assessment, opportunities section now renders three tiers,
     each linking to its source application(s)
```

Persisted, unlike the Q&A summary (which is stateless by design — no memory of prior questions).
Opportunities differ: they belong to *this specific assessment*, same as every other field, so a
second viewer of the same report shouldn't have to regenerate them, and the export (docx/pdf)
needs real content to include rather than nothing. This is the one deliberate divergence from the
Q&A precedent, and it's why an endpoint (not a stateless call) plus a `potential_opportunities`
write path is needed, not just a raw passthrough like `/api/ask/summarize`.

## Backend

**New module** `src/researchbridge/assessment/opportunity_synthesis.py` (kept separate from
`opportunities.py`'s stable `assess_opportunities()` interface point — that function's signature is
reused as the entry point once this lands, per its own docstring's stated intent):

```python
@dataclass
class SynthesizedOpportunity:
    tier: Literal["direct", "adjacent", "speculative"]
    opportunity: str
    source_application_indices: list[int]  # 1-indexed into the applications list given

@dataclass
class SynthesisResult:
    opportunities: list[SynthesizedOpportunity]

class OpportunitySynthesisUnavailable(Exception):
    """Raised when OLLAMA_ENABLED is false, no applications exist to ground
    synthesis in, Ollama is unreachable, or validation fails after retry.
    Route layer turns this into a 503 (or 422 for the no-applications case)."""

def build_prompt(applications: list[ApplicationRecord]) -> tuple[str, str]: ...
def parse_response(text: str, application_count: int) -> list[SynthesizedOpportunity]:
    """Raises ValueError if not exactly one Direct, one Adjacent, and one
    Speculative line are found, or any cited index is out of range."""
def synthesize_opportunities(applications: list[ApplicationRecord]) -> SynthesisResult: ...
```

**Prompt design** (system prompt):

> "You are given a numbered list of applications already identified for a research idea, each
> grounded in a specific paper. Propose exactly three product/technology opportunities that build
> on these applications: one Direct (a straightforward product built from one application as
> stated), one Adjacent (a broader product combining or extending the applications, still
> plausible from what's listed), and one Speculative (an ambitious, longer-horizon idea, clearly
> still connected to the applications). Do not invent a capability, technology, or claim that isn't
> implied by the numbered applications. Format your response as exactly three lines:
> 'Direct: <opportunity> [n]', 'Adjacent: <opportunity> [n][m]', 'Speculative: <opportunity> [n]' —
> each line's [n] citing which application number(s) it draws from."

Temperature ~0.2, same as the Q&A layer. Reuses `_call_ollama`-equivalent logic — likely worth
factoring `_call_ollama` out of `qa/summarize.py` into a small shared helper (e.g.
`researchbridge/llm/ollama.py`) at implementation time, so this module and `qa/summarize.py` don't
duplicate the HTTP-call/retry-envelope code; the prompt-building and response-parsing stay
separate per-feature, since the required output shape differs (free-text-with-citations vs.
three-labeled-tiers-with-citations).

**No new config** — reuses `OLLAMA_ENABLED`/`OLLAMA_MODEL`/`OLLAMA_HOST`/`OLLAMA_TIMEOUT_SECONDS`
from `qa/summarize.py` as-is.

**Route** (new, in `src/researchbridge/api/assessment_routes.py`):
```
POST /api/assessments/{id}/opportunities
  404 if the assessment doesn't exist
  422 if potential_applications is empty/null — nothing to ground synthesis in
  503 if OLLAMA_ENABLED is false, or Ollama is unreachable/times out, or validation
      fails after one retry
  200: the updated ResearchAssessmentOut, with potential_opportunities populated
```

**Schema**: `potential_opportunities: list[dict] | None` (existing field, `models.py:464`,
already `JSONB`, no migration needed) — each dict shaped as
`{"tier": "direct"|"adjacent"|"speculative", "opportunity": str, "source_applications": [{"application": str, "paper_id": str, "paper_title": str}, ...]}`,
resolved server-side from `source_application_indices` before persisting, so the frontend never
has to re-derive which application a citation index pointed to.

## Frontend

**`frontend/lib/assessmentApi.ts`**: add `synthesizeOpportunities: (id: string) =>
Promise<ResearchAssessment>`, and extend the `PotentialApplication`-adjacent types with the
`SynthesizedOpportunity` shape above (replacing the current untyped `unknown[] | null` for
`potential_opportunities`).

**`frontend/components/AssessmentReport.tsx`** — the "product / technology opportunities" `Field`
(currently a permanent `Unassessed` placeholder, lines 183–185):
- If `potential_applications` is empty/null: keep today's message unchanged (nothing to ground
  synthesis in — no button shown).
- Else if `potential_opportunities` is null: show a "✨ synthesize opportunities" button, same
  visual language as the Q&A summary's button, labeled so it's clearly optional/AI-generated
  before it's clicked.
- Else: render the three tiers (Direct/Adjacent/Speculative), each linking to its source
  application(s)/paper(s), with a persistent "AI-synthesized — not independently verified" label
  matching the Q&A summary panel's framing.
- On 503/failure: inline error, existing report content untouched.

## Error handling

- `OLLAMA_ENABLED=false` (default): section shows today's unchanged message; no button rendered;
  calling the route directly would 503.
- `potential_applications` empty or null: 422, button never shown — this is a permanent state for
  that assessment (re-running the assessment could change it if new evidence appears, but clicking
  won't).
- Ollama unreachable/times out, or the response doesn't parse into exactly three cited tiers:
  retry once, then 503. `potential_opportunities` stays `NULL` — never a partial (e.g. two tiers)
  result persisted.
- Once successfully synthesized and persisted, the button is replaced by the rendered result for
  every future view of that assessment (no re-synthesis loop, no re-hitting Ollama on every page
  load).

## Testing

Following this codebase's own established split (see `qa/summarize.py` / `test_qa_summarize.py`):
- `build_prompt`: numbering matches `potential_applications` order — pure function.
- `parse_response`: all three tiers present + valid citations accepted; missing a tier, malformed
  tier label, out-of-range citation, and duplicate tier all raise `ValueError` — pure function, no
  Ollama needed, exhaustive table of malformed-response shapes.
- `synthesize_opportunities`: `requests.post` mocked — happy path, retry-then-succeed,
  retry-then-fail-closed, timeout, disabled.
- Route tests (`test_assessment_api.py` additions): 200 with valid synthesis (mocked Ollama), 422
  when no applications exist, 503 when disabled, 503 when Ollama mocked unreachable, verify
  `potential_opportunities` persists correctly and survives a subsequent plain `GET`.
- Frontend: this repo's Vitest/RTL infrastructure now exists (see recent test-infra work) — add a
  render test confirming the button appears only when applications exist and opportunities are
  still null, and that a populated `potential_opportunities` renders all three tiers with working
  source links.

## Migration safety

No schema change — `potential_opportunities` (`JSONB`, nullable) already exists on
`research_assessments`. No change to `build_assessment()` or any existing `assess_*` function's
behavior; `assess_opportunities()` keeps returning `NULL` for every assessment until this feature
is explicitly triggered per-assessment via the new route. Fully additive and inert unless
`OLLAMA_ENABLED=true` is already set (an operator choice already made for the Q&A layer, or made
fresh for this one) **and** a user explicitly clicks the button.

## Open questions for whoever implements this

1. Should a failed synthesis attempt be silently retryable by clicking the button again, or should
   the UI show a distinct "last attempt failed at <time>" state? (The Q&A layer doesn't need this —
   it's stateless per-question. This feature persists, so a failed attempt is more visible.)
2. Blueprint Sec 33's worked example implies one opportunity per tier. Is that a hard constraint,
   or should the model be allowed to propose e.g. two Adjacent opportunities if the applications
   support it? This spec assumes exactly one per tier (simpler prompt, simpler validation,
   matches the worked example precisely) — worth confirming against Sec 33/49's intent before
   building `parse_response`'s validation around it.
3. Worth exposing `OLLAMA_MODEL` per-feature (a smaller/faster model for the Q&A layer, a
   different one for opportunities) rather than sharing one setting? This spec shares it for
   simplicity; revisit if quality tradeoffs turn out to differ meaningfully between the two tasks.
