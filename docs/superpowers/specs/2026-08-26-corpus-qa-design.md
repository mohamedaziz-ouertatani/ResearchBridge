# Corpus Q&A ("Ask the corpus") — Design Spec

Source of truth for architecture: `ResearchBridge.md` (blueprint). This is the RAG feature from the product backlog: a way to ask a free-text question and get back grounded evidence from the corpus, without introducing generation where the codebase has deliberately avoided it everywhere else (extraction/heuristic.py, semantic.py, hybrid.py all take claim text verbatim from source sentences specifically so the grounding check — an exact-substring test — always passes; see Evidence's "no Grounding Illusion" framing).

## Scope

In scope:
- One new endpoint, `POST /api/ask`, that takes a free-text question and returns a ranked list of real, already-extracted quotes (claim text + its grounding Evidence) from the corpus, each citing its source paper.
- One new standalone page, `/ask`: a question box, and the ranked quote results below it.
- Two-stage retrieval: paper-level embedding search (reusing `search_by_text`) narrows to a candidate set, then a query-time re-rank of those candidates' claim/evidence text against the exact question surfaces the specific relevant quotes.
- Respecting the two exclusion rules already enforced everywhere else in the corpus: `Paper.excluded_at IS NULL` (curated-out papers never surface) and `extraction_method != "stub"` (synthetic placeholder data is never treated as real evidence).

Out of scope (deliberately deferred):
- Any generative LLM call. No new dependency, no API key, no free-text synthesis — the "answer" is a set of real quotes, not prose written about them. This is the load-bearing decision from brainstorming; revisiting it is a separate, later decision if ever made.
- Full-text search over each paper's whole body. The general corpus stores only title+abstract (`Paper` has no fulltext column — see `benchmark/fulltext.py`'s docstring, which caches full text only for the ~40 benchmark papers, deliberately not wired into the main pipeline). This feature searches the same ground truth every other corpus feature does: abstracts plus whatever `ExtractedClaim`/`Evidence` the extraction pipeline already pulled out.
- Persisting questions or answers anywhere. Stateless request/response, like `GET /api/search` — no new table, no history page.
- A new per-claim embedding column/pipeline job. Candidate quotes are embedded on the fly, at query time, over the small set surfaced by the paper-level search (bounded — see Performance below).
- Multi-turn conversation / follow-up context. One question in, one ranked result set out, same as a search box.

## Data flow

```
question (free text)
  -> embed with the existing SentenceTransformerEmbedder singleton (get_embedder())
  -> search_by_text(): pgvector cosine search over Paper embeddings (title+abstract),
     top ~10 candidate papers, excluded_at IS NULL (existing filter, unchanged)
  -> for each candidate paper: load its ExtractedClaim + Evidence rows
     where evidence.extraction_method != "stub"
  -> embed that batch of quote texts (typically well under 100 short strings -
     a handful of claim fields per paper x ~10 papers) with the same embedder
  -> cosine-rank quotes against the question embedding, take top ~8
  -> return them ranked, each citing its paper (id, title, source) and claim_type
```

No new embedding pipeline, no new migration. The only new computation is embedding a small, bounded, per-request batch of already-short strings (claim/evidence text is at most a sentence or two) — reusing the same CPU-friendly MiniLM model every other embedding call in this app already uses.

## Backend

**New module** `src/researchbridge/qa/answer.py`:

```python
@dataclass
class QuoteHit:
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float  # cosine similarity to the question, for display/sort only

def answer_question(
    session: Session, embedder: Embedder, question: str,
    top_k_papers: int = 10, top_k_quotes: int = 8,
) -> list[QuoteHit]:
    ...
```

Reuses `embedding/search.py::search_by_text` for stage one exactly as-is (already respects `excluded_at`). Stage two is new: load candidates' non-stub `Evidence`/`ExtractedClaim` pairs (join, same shape as `assessment/gap.py`'s existing evidence-loading queries), embed their text via `embedder.embed_texts(...)`, rank by cosine similarity against the question's own embedding (plain dot product — vectors are already L2-normalized, matching the convention documented on `Embedding.vector`).

**New schemas** in `api/schemas.py`:
```python
class AskRequest(BaseModel):
    question: str

class QuoteHitOut(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float

class AskResponse(BaseModel):
    hits: list[QuoteHitOut]
```

**New route file** `src/researchbridge/api/qa_routes.py`:
```
POST /api/ask
  body: {"question": str}
  422 if question is empty/whitespace-only (mirrors ResearchAssessmentCreate's raw_text validation)
  200: AskResponse (hits: [] when no candidate paper has any real evidence - never a 404,
       empty results is a valid answer to "nothing grounded exists for this yet")
```

Registered in `app.py` alongside the other routers. Uses the existing `get_session`/`get_embedder` dependencies — no new ones.

## Frontend

**New page** `frontend/app/ask/page.tsx`:
- Header matching the existing page pattern (`← ResearchBridge` link + page label).
- A single-line question input + submit button, styled like the homepage's idea textarea (`app/page.tsx`'s `<textarea>` block) but single-line since a question is normally short.
- Below: a list of quote cards, each showing the quote text, an eyebrow badge for `claim_type`, the section (if any), a link to `/papers/{paper_id}`, and the paper's title/source. No result count/relevance score shown numerically (matches the app's existing convention of not surfacing raw confidence/score numbers to end users outside the admin panel) — order alone conveys rank.
- Empty state: "No grounded evidence found for this question yet — try rephrasing, or the corpus may not have extracted claims covering this topic."
- Loading state: same busy/disabled-button pattern used by the homepage's assessment submission.

**New API client** `frontend/lib/qaApi.ts`:
```ts
export type QuoteHit = { paper_id: string; paper_title: string; paper_source: string;
  claim_type: string; text: string; section: string | null; confidence: string; score: number };
export const qaApi = { ask: (question: string) => Promise<{ hits: QuoteHit[] }> };
```

**Nav entry**: one link added to the homepage console's existing nav row (`app/page.tsx`, alongside `assessments →`, `corpus →`, `gap review →`, `annotation workbench →`, `pipeline status →`) — `ask the corpus →`.

## Error handling

- Empty/whitespace question → 422 from the backend (same shape as existing validation errors), surfaced as an inline form error on `/ask`, no request sent for an obviously-empty input (client-side guard mirrors the homepage's `idea.trim()` check).
- Zero candidate papers (empty corpus) or zero non-stub evidence among candidates → 200 with `hits: []`, rendered as the empty state above. Never falls back to raw abstract text dressed up as an "answer" — that would violate the same grounding guarantee the rest of the app maintains.
- API unreachable → same inline error pattern already used on `/`, `/assessments`, `/corpus` ("Can't reach the API...").

## Testing

- Backend: pytest unit tests for `answer_question()` — ranks quotes by relevance, respects `excluded_at`, filters `extraction_method == "stub"`, returns `[]` when no candidate has real evidence, handles a corpus with zero papers. Plus route-level tests (`test_qa_api.py`, following `test_api.py`'s fixture/style) — 200 with results, 200 with empty `hits`, 422 on empty question.
- Frontend: no test infra in this repo (consistent with every other feature this session) — verified live via browser-preview: ask a real question against the live corpus, confirm quotes render with working paper links, confirm the empty state renders for a nonsense query, confirm the nav link works from the homepage.

## Migration safety

No migration. No schema change. Purely additive: one new route file, one new frontend page, zero changes to any existing table, endpoint, or page's behavior.
