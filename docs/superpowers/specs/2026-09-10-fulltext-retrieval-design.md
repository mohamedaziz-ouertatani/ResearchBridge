# Full-text retrieval for "Ask the corpus" — Design Spec

Source of truth for prior state: `docs/superpowers/specs/2026-08-26-corpus-qa-design.md` (original two-stage retrieval design, deliberately abstract-only at the time — see its "Out of scope" section, item "Full-text search over each paper's whole body"). This spec revisits that deferral now that `PaperFullText` (`src/researchbridge/db/models.py:512`) exists and is populated for open-access papers by `src/researchbridge/fulltext/pipeline.py`.

This is sub-project 1 of a three-part RAG enhancement (full-text retrieval → scale/performance → answer synthesis quality), scoped and ordered during brainstorming. It intentionally does not touch ANN indexing or the Ollama summarization layer.

## Scope

In scope:
- Chunk each open-access paper's `PaperFullText.sections` into paragraphs and embed them (new table, new pipeline job).
- Widen Stage 1 (paper candidate selection) to also consider full-text paragraph similarity, not just title/abstract similarity.
- Widen Stage 2 (quote selection) to surface raw full-text paragraphs as citable "passages," alongside the existing extracted claim/evidence "claims," deduplicated against each other.
- Additive API/frontend changes to distinguish `"claim"` vs `"passage"` hits.

Out of scope (deferred to later sub-projects or explicitly rejected):
- ANN indexing (ivfflat/hnsw) for the new chunk table — sub-project 2. Stage 1's new per-paper best-chunk-score subquery will do a sequential scan, same accepted tradeoff as the existing abstract search (`src/researchbridge/embedding/search.py:1-6`).
- Persisting/caching quote embeddings — out of scope here; this spec adds a *new persisted* embedding column (chunks), which is a separate concern from the existing on-the-fly claim/evidence embedding described in the original design (item still holds for claims).
- Any change to the Ollama summarization layer (`src/researchbridge/qa/summarize.py`) — sub-project 3. It receives whatever hits Stage 2 returns, unchanged.
- Papers without a `PaperFullText` row (non-open-access, or not yet fetched): retrieval falls back to exactly today's behavior (abstract + claims only) for those papers. No new extraction work.
- Multi-turn/follow-up context — still one question in, one result set out.

## Data flow

```
question (free text)
  -> embed with existing SentenceTransformerEmbedder singleton
  -> Stage 1: candidate papers = top-10 by max(
       abstract_embedding_score,          # existing Paper-embedding cosine search
       best_chunk_score                   # NEW: max cosine sim across this paper's
     )                                    #      PaperFullTextChunk rows (GROUP BY paper_id)
     filtered by excluded_at IS NULL (unchanged)
  -> Stage 2 candidate pool, per candidate paper:
       - existing ExtractedClaim + Evidence rows (extraction_method != "stub")
       - NEW: PaperFullTextChunk paragraphs for that paper
  -> dedupe: drop a passage chunk whose text contains an already-present claim's
     quote text verbatim (claims are literal substrings of some paragraph)
  -> embed claim texts on the fly (unchanged); chunk texts use their persisted embeddings
  -> cosine-rank the merged pool against the question embedding, take top ~8
  -> return them ranked, each tagged source="claim" | "passage", citing paper + section
```

## Backend

**New table** (migration required), `PaperFullTextChunk`:

```python
class PaperFullTextChunk(Base):
    __tablename__ = "paper_fulltext_chunk"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id"), index=True)
    section: Mapped[str] = mapped_column(String)
    paragraph_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
```

Same embedding convention as `Embedding` (pgvector, 384-dim MiniLM, L2-normalized — see `embedding/model.py`).

**New module** `src/researchbridge/fulltext/chunking.py`:

```python
def split_paragraphs(section_text: str, min_words: int = 10) -> list[str]:
    """Split on blank-line boundaries; merge any paragraph under min_words
    into the following one (drops stray headers/fragments as standalone chunks)."""
```

**New pipeline job**, mirroring the existing paper-embedding pipeline shape (`src/researchbridge/embedding/pipeline.py`): for each `PaperFullText` row without matching `PaperFullTextChunk` rows (or where `PaperFullText` was updated more recently than its chunks), split each section into paragraphs via `split_paragraphs`, embed each with the shared `SentenceTransformerEmbedder`, insert one `PaperFullTextChunk` row per paragraph. Runs as a batch backfill initially, then incrementally whenever `FullTextFetchPipeline` persists a new/updated `PaperFullText` row (same trigger pattern already used for paper-level embeddings).

**`src/researchbridge/embedding/search.py`**: add a `best_chunk_score` helper — a `GROUP BY paper_id, MAX(cosine_score)` query over `PaperFullTextChunk` — and combine it with the existing abstract-score query in `search_by_text` (or a thin wrapper) via `GREATEST()` per paper, keeping the existing top-10 cutoff and `excluded_at IS NULL` filter.

**`src/researchbridge/qa/answer.py`**: `QuoteHit` gains a `source: Literal["claim", "passage"]` field. Stage 2 loads `PaperFullTextChunk` rows for candidate papers alongside the existing claim/evidence query, applies the substring-dedupe rule, ranks the merged pool by cosine similarity (chunk embeddings already persisted — no on-the-fly embedding needed for them; claim texts still embedded on the fly as today).

**Schemas** (`api/schemas.py`): `QuoteHitOut` gains `source: Literal["claim", "passage"]`. Additive, backward-compatible.

## Frontend

`frontend/app/ask/page.tsx`'s `QuoteCard`: when `source === "passage"`, render a "from full text" label instead of the claim-type badge (passages weren't vetted by claim extraction, so they shouldn't look claim-typed). No other structural change; existing citation-link and collection-save behavior is unaffected since it already keys off `paper_id`/hit index, not `source`.

## Rollout

1. Migration adds `paper_fulltext_chunk` table.
2. Backfill job runs once over all existing `PaperFullText` rows (row-count/volume should be sanity-checked before running — no hard cap in this spec, since exact open-access coverage of the ~47K corpus is undetermined; if backfill proves too large for a single run, it can be batched by paper without further design changes).
3. Retrieval code changes ship behind no flag — same as the original Q&A feature, this is additive and fails soft (papers without chunks behave exactly as before).

## Testing

- `fulltext/chunking.py`: paragraph splitting, short-paragraph merging, empty-section handling, section with only headers.
- `qa/answer.py`: merged-pool ranking includes both sources, dedupe drops a passage that contains an already-selected claim's quote, papers without `PaperFullText` behave identically to current tests, `source` field correctness.
- `embedding/search.py`: `best_chunk_score` combination with abstract score (paper with only chunk match still surfaces, paper with only abstract match still surfaces, paper with both takes the higher).
- Route-level: `/api/ask` response includes correctly tagged `source` for a paper with full text vs. one without.

## Open questions (none blocking — noted for implementation time)

- Exact fraction of the corpus with `PaperFullText` rows is unknown; worth a quick count before backfill to gauge chunk-table row volume and pipeline runtime.
