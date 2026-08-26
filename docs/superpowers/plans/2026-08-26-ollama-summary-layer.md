# Local-LLM Summarization Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, off-by-default local-LLM summarization layer on top of the existing extractive `/api/ask`, so a user can request a short synthesized answer (with citations) built only from the already-retrieved quotes.

**Architecture:** `POST /api/ask` stays unchanged except for one new response field (`summarization_available`). A new `POST /api/ask/summarize` endpoint accepts the question plus the exact hits the client already has, calls a local Ollama model with a constrained prompt, and validates every citation marker in the response against the hits it was given before returning. The frontend calls this only when a user clicks a button that appears after quotes render; quotes are never blocked, replaced, or altered.

**Tech Stack:** FastAPI, Pydantic, `requests` (existing dependency, matches `connectors/springer.py`/`connectors/semantic_scholar.py`), Ollama's local HTTP API (`/api/chat`) at `localhost:11434`, Next.js/React (existing `frontend/app/ask/page.tsx`).

**Spec:** [docs/superpowers/specs/2026-08-26-ollama-summary-layer-design.md](../specs/2026-08-26-ollama-summary-layer-design.md)

## Global Constraints

- `POST /api/ask` behavior and response fields (other than the one addition) are unchanged — byte-for-byte compatible.
- The feature is off unless `OLLAMA_ENABLED=true` is set; default is `false`.
- Default model: `qwen2.5:3b` (already installed locally). Configurable via `OLLAMA_MODEL`.
- Default `OLLAMA_HOST`: `http://localhost:11434` (Ollama's default port).
- Default `OLLAMA_TIMEOUT_SECONDS`: `30`.
- No new Python dependency — use `requests`, already used elsewhere in this codebase.
- No new database table, no migration.
- Exactly one retry on invalid citations or a failed Ollama call; after that, fail closed (raise/503) — never return a summary with an unverified citation.
- Raw quotes must remain visible and unaltered on `/ask` at all times, regardless of summarization state.

---

### Task 1: Prompt building and citation extraction (pure functions)

**Files:**
- Create: `src/researchbridge/qa/summarize.py`
- Test: `tests/test_qa_summarize.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT: str` (module constant), `build_prompt(question: str, hits: list[QuoteHitOut]) -> tuple[str, str]`, `extract_citations(text: str, hit_count: int) -> list[int]`.
- Consumes: `QuoteHitOut` from `researchbridge.api.schemas` (existing pattern — `assessment/export.py` already imports schemas from the api layer, see [export.py:22](../../src/researchbridge/assessment/export.py)).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_qa_summarize.py
from __future__ import annotations

import uuid

import pytest

from researchbridge.api.schemas import QuoteHitOut
from researchbridge.qa.summarize import build_prompt, extract_citations


def _hit(text: str, paper_title: str = "Some Paper") -> QuoteHitOut:
    return QuoteHitOut(
        paper_id=uuid.uuid4(),
        paper_title=paper_title,
        paper_source="arxiv",
        claim_type="limitations",
        text=text,
        section=None,
        confidence="medium",
        score=0.9,
    )


def test_build_prompt_numbers_hits_in_order() -> None:
    hits = [_hit("first quote", "Paper A"), _hit("second quote", "Paper B")]

    system_prompt, user_prompt = build_prompt("what are the limitations?", hits)

    assert "only use" in system_prompt.lower() or "only" in system_prompt.lower()
    assert '[1] "first quote" — Paper A' in user_prompt
    assert '[2] "second quote" — Paper B' in user_prompt
    assert "what are the limitations?" in user_prompt


def test_extract_citations_returns_unique_numbers_in_order_of_appearance() -> None:
    text = "Models struggle offline [2]. This was also noted elsewhere [1][2]."

    citations = extract_citations(text, hit_count=2)

    assert citations == [2, 1]


def test_extract_citations_returns_empty_list_when_no_citations_present() -> None:
    citations = extract_citations("The quotes don't address this question.", hit_count=3)

    assert citations == []


def test_extract_citations_raises_on_out_of_range_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("This claims something [5].", hit_count=2)


def test_extract_citations_raises_on_zero_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("Cites [0] which isn't a valid index.", hit_count=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_summarize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.qa.summarize'`

- [ ] **Step 3: Write the implementation**

```python
# src/researchbridge/qa/summarize.py
"""Local-LLM summarization layer over the extractive Q&A quotes.

Takes the exact QuoteHitOut list the client already received from
POST /api/ask and asks a local Ollama model to write a short synthesis
that only rephrases/connects those quotes, citing each by its [n]
index. Never invents new facts by design - the prompt constrains the
model to the numbered quotes, and every citation the model emits is
checked against the hits it was actually given (see extract_citations).
This is the one place in the app that calls a generative model; see
docs/superpowers/specs/2026-08-26-ollama-summary-layer-design.md for
why, and why it stays optional/off-by-default.
"""

from __future__ import annotations

import re

from researchbridge.api.schemas import QuoteHitOut

SYSTEM_PROMPT = (
    "You are given a question and a numbered list of verbatim quotes from research papers. "
    "Write a 3-5 sentence synthesis using ONLY information stated in these quotes - never add "
    "outside knowledge or infer anything not explicitly present. After every sentence, cite the "
    "quote number(s) it draws from in brackets, e.g. [1] or [1][3]. If the quotes don't address "
    "the question, say that plainly instead of guessing. Do not use any information not in the "
    "numbered quotes above - only use the quotes given."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, hits: list[QuoteHitOut]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the Ollama chat call."""
    numbered = "\n".join(
        f'[{i}] "{hit.text}" — {hit.paper_title}' for i, hit in enumerate(hits, start=1)
    )
    user_prompt = f"Question: {question}\n\nQuotes:\n{numbered}"
    return SYSTEM_PROMPT, user_prompt


def extract_citations(text: str, hit_count: int) -> list[int]:
    """Extracts [n] citation markers from the model's output, in order of
    first appearance, deduplicated. Raises ValueError if any citation
    number is outside 1..hit_count - the caller treats this identically
    to an unreachable model (retry, then fail closed)."""
    seen: list[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        n = int(match.group(1))
        if n < 1 or n > hit_count:
            raise ValueError(f"citation [{n}] is out of range for {hit_count} quotes")
        if n not in seen:
            seen.append(n)
    return seen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qa_summarize.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/qa/summarize.py tests/test_qa_summarize.py
git commit -m "feat: add prompt building and citation extraction for Ollama summary layer"
```

---

### Task 2: Ollama call orchestration with retry and fail-closed behavior

**Files:**
- Modify: `src/researchbridge/qa/summarize.py`
- Test: `tests/test_qa_summarize.py`

**Interfaces:**
- Consumes: `build_prompt`, `extract_citations`, `SYSTEM_PROMPT` from Task 1 (same module).
- Produces: `SummaryResult` dataclass (`summary: str`, `citations: list[int]`), `SummarizationUnavailable(Exception)`, `ollama_enabled() -> bool`, `summarize_quotes(question: str, hits: list[QuoteHitOut]) -> SummaryResult` — all consumed by Task 3's route.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qa_summarize.py`:

```python
from unittest.mock import Mock

from researchbridge.qa.summarize import (
    SummarizationUnavailable,
    ollama_enabled,
    summarize_quotes,
)


def _mock_ollama_response(monkeypatch: pytest.MonkeyPatch, content: str) -> Mock:
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": content}}
    mock_response.raise_for_status = Mock()
    mock_post = Mock(return_value=mock_response)
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)
    return mock_post


def test_ollama_enabled_reflects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    assert ollama_enabled() is True

    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    assert ollama_enabled() is False

    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)
    assert ollama_enabled() is False


def test_summarize_quotes_raises_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    hits = [_hit("some quote")]

    with pytest.raises(SummarizationUnavailable, match="not enabled"):
        summarize_quotes("a question", hits)


def test_summarize_quotes_returns_validated_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("first quote"), _hit("second quote")]
    _mock_ollama_response(monkeypatch, "This is grounded [1] and also this [2].")

    result = summarize_quotes("a question", hits)

    assert result.summary == "This is grounded [1] and also this [2]."
    assert result.citations == [1, 2]


def test_summarize_quotes_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    mock_response_bad = Mock()
    mock_response_bad.json.return_value = {"message": {"content": "Invalid cite [9]."}}
    mock_response_bad.raise_for_status = Mock()
    mock_response_good = Mock()
    mock_response_good.json.return_value = {"message": {"content": "Valid cite [1]."}}
    mock_response_good.raise_for_status = Mock()
    mock_post = Mock(side_effect=[mock_response_bad, mock_response_good])
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)

    result = summarize_quotes("a question", hits)

    assert result.summary == "Valid cite [1]."
    assert mock_post.call_count == 2


def test_summarize_quotes_fails_closed_after_two_bad_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    _mock_ollama_response(monkeypatch, "Always invalid [9].")

    with pytest.raises(SummarizationUnavailable, match="valid grounded summary"):
        summarize_quotes("a question", hits)


def test_summarize_quotes_fails_closed_when_ollama_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    hits = [_hit("only quote")]
    mock_post = Mock(side_effect=requests.ConnectionError("connection refused"))
    monkeypatch.setattr("researchbridge.qa.summarize.requests.post", mock_post)

    with pytest.raises(SummarizationUnavailable, match="valid grounded summary"):
        summarize_quotes("a question", hits)

    assert mock_post.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_summarize.py -v`
Expected: FAIL with `ImportError: cannot import name 'SummarizationUnavailable'`

- [ ] **Step 3: Write the implementation**

Add to `src/researchbridge/qa/summarize.py` (after the `extract_citations` function, add the `os`/`requests`/`dataclasses` imports at the top):

```python
# add to imports at top of the file
import os
from dataclasses import dataclass

import requests
```

```python
@dataclass
class SummaryResult:
    summary: str
    citations: list[int]


class SummarizationUnavailable(Exception):
    """Raised when OLLAMA_ENABLED is false, or Ollama is unreachable/times
    out/produces an invalid summary after one retry. The route layer turns
    this into a 503 - never a partially-validated summary."""


def ollama_enabled() -> bool:
    return os.environ.get("OLLAMA_ENABLED", "false").lower() == "true"


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30"))

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def summarize_quotes(question: str, hits: list[QuoteHitOut]) -> SummaryResult:
    """Calls the local Ollama model to synthesize a grounded summary of
    the given hits. Retries once on an unreachable model or an
    out-of-range citation, then raises SummarizationUnavailable - never
    returns a summary whose citations weren't checked against hits."""
    if not ollama_enabled():
        raise SummarizationUnavailable("local LLM summarization is not enabled")

    system_prompt, user_prompt = build_prompt(question, hits)

    for _attempt in range(2):
        try:
            content = _call_ollama(system_prompt, user_prompt)
            citations = extract_citations(content, len(hits))
        except (requests.RequestException, ValueError, KeyError):
            continue
        return SummaryResult(summary=content, citations=citations)

    raise SummarizationUnavailable("local LLM could not produce a valid grounded summary")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qa_summarize.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/qa/summarize.py tests/test_qa_summarize.py
git commit -m "feat: orchestrate Ollama call with retry and fail-closed validation"
```

---

### Task 3: Schemas, route, and config wiring

**Files:**
- Modify: `src/researchbridge/api/schemas.py`
- Modify: `src/researchbridge/api/qa_routes.py`
- Modify: `.env.example`
- Test: `tests/test_qa_api.py`

**Interfaces:**
- Consumes: `summarize_quotes`, `SummarizationUnavailable`, `ollama_enabled` from `researchbridge.qa.summarize` (Task 2); `QuoteHitOut`, `AskResponse` from `researchbridge.api.schemas`.
- Produces: `POST /api/ask/summarize` route; `AskResponse.summarization_available: bool` field, consumed by the frontend in Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qa_api.py`:

```python
def test_ask_reports_summarization_available_true_when_enabled(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.json()["summarization_available"] is True


def test_ask_reports_summarization_available_false_by_default(client) -> None:
    response = client.post("/api/ask", json={"question": "anything"})

    assert response.json()["summarization_available"] is False


def test_summarize_returns_summary_when_enabled(
    client, session, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_response = Mock()
    mock_response.json.return_value = {"message": {"content": "Grounded summary [1]."}}
    mock_response.raise_for_status = Mock()
    monkeypatch.setattr(
        "researchbridge.qa.summarize.requests.post", Mock(return_value=mock_response)
    )
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Grounded summary [1]."
    assert body["citations"] == [1]


def test_summarize_returns_503_when_disabled(client) -> None:
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 503


def test_summarize_returns_503_when_ollama_unreachable(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    monkeypatch.setattr(
        "researchbridge.qa.summarize.requests.post",
        Mock(side_effect=requests.ConnectionError("refused")),
    )
    hit = {
        "paper_id": str(uuid.uuid4()),
        "paper_title": "Paper A",
        "paper_source": "arxiv",
        "claim_type": "limitations",
        "text": "some quote",
        "section": None,
        "confidence": "medium",
        "score": 0.9,
    }

    response = client.post(
        "/api/ask/summarize", json={"question": "a question", "hits": [hit]}
    )

    assert response.status_code == 503


def test_summarize_rejects_empty_hits(client) -> None:
    response = client.post("/api/ask/summarize", json={"question": "a question", "hits": []})

    assert response.status_code == 422
```

Add these imports at the top of `tests/test_qa_api.py` alongside the existing ones:

```python
from unittest.mock import Mock
```

(`uuid` and `pytest` are already imported in this file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_api.py -v`
Expected: FAIL — `summarization_available` KeyError / 404 on `/api/ask/summarize` (route doesn't exist yet)

- [ ] **Step 3: Write the implementation**

In `src/researchbridge/api/schemas.py`, replace the existing `AskResponse` definition:

```python
class AskResponse(BaseModel):
    hits: list[QuoteHitOut]
    summarization_available: bool
```

Add two new schemas directly below it:

```python
class SummarizeRequest(BaseModel):
    question: str = Field(min_length=1)
    hits: list[QuoteHitOut] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    citations: list[int]
```

Rewrite `src/researchbridge/api/qa_routes.py` in full:

```python
"""Extractive Q&A over the corpus (blueprint RAG slice), plus an optional
local-LLM summarization layer.

POST /api/ask retrieves candidate papers by embedding similarity, then
re-ranks their already-extracted claims/evidence against the question -
see qa/answer.py's module docstring for why this is retrieval, not
generation. That endpoint's behavior is unchanged by the layer below.

POST /api/ask/summarize is the optional layer: it takes the exact hits
a client already received from /api/ask and asks a local Ollama model
to synthesize a short, cited summary of them - see qa/summarize.py's
module docstring and docs/superpowers/specs/
2026-08-26-ollama-summary-layer-design.md for the grounding guarantees.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    AskRequest,
    AskResponse,
    QuoteHitOut,
    SummarizeRequest,
    SummarizeResponse,
)
from researchbridge.embedding.base import Embedder
from researchbridge.qa.answer import answer_question
from researchbridge.qa.summarize import SummarizationUnavailable, ollama_enabled, summarize_quotes

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> AskResponse:
    hits = answer_question(session, embedder, payload.question)
    return AskResponse(
        hits=[
            QuoteHitOut(
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                paper_source=hit.paper_source,
                claim_type=hit.claim_type,
                text=hit.text,
                section=hit.section,
                confidence=hit.confidence,
                score=hit.score,
            )
            for hit in hits
        ],
        summarization_available=ollama_enabled(),
    )


@router.post("/ask/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    try:
        result = summarize_quotes(payload.question, payload.hits)
    except SummarizationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SummarizeResponse(summary=result.summary, citations=result.citations)
```

Add to `.env.example` (after the `SEMANTIC_SCHOLAR_API_KEY` block):

```
# Optional - local-LLM summarization layer on top of the extractive /api/ask
# results. Off by default. Requires a running Ollama server
# (https://ollama.com) with OLLAMA_MODEL pulled, e.g. `ollama pull qwen2.5:3b`.
OLLAMA_ENABLED=false
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=30
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qa_api.py tests/test_qa_summarize.py -v`
Expected: PASS (all tests, including the pre-existing `/api/ask` tests — confirm none regressed)

- [ ] **Step 5: Run the full backend test suite**

Run: `pytest -v`
Expected: PASS (no regressions anywhere else in the suite)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/qa_routes.py .env.example tests/test_qa_api.py
git commit -m "feat: add POST /api/ask/summarize endpoint and summarization_available flag"
```

---

### Task 4: Frontend API client

**Files:**
- Modify: `frontend/lib/qaApi.ts`

**Interfaces:**
- Consumes: `SummarizeResponse` shape `{summary: string, citations: number[]}` from Task 3's route.
- Produces: `qaApi.summarize(question, hits) -> Promise<{summary: string; citations: number[]}>`, `QuoteHit`/`AskResponse` types extended, consumed by Task 5's page component.

- [ ] **Step 1: Update the API client**

Rewrite `frontend/lib/qaApi.ts` in full:

```ts
import { API_BASE } from "./api";

export type QuoteHit = {
  paper_id: string;
  paper_title: string;
  paper_source: string;
  claim_type: string;
  text: string;
  section: string | null;
  confidence: string;
  score: number;
};

export type AskResponse = {
  hits: QuoteHit[];
  summarization_available: boolean;
};

export type SummarizeResponse = {
  summary: string;
  citations: number[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

export const qaApi = {
  ask: (question: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  summarize: (question: string, hits: QuoteHit[]) =>
    request<SummarizeResponse>("/api/ask/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, hits }),
    }),
};
```

- [ ] **Step 2: Verify the frontend typechecks**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new type errors (existing `AskResponse.hits` usages in `app/ask/page.tsx` still compile since `hits` is untouched; `summarization_available` is unused there until Task 5)

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/qaApi.ts
git commit -m "feat: add summarize() to the QA API client"
```

---

### Task 5: Frontend UI — summary panel, citation links, highlight animation

**Files:**
- Modify: `frontend/app/ask/page.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `qaApi.summarize`, `AskResponse.summarization_available`, `SummarizeResponse` from Task 4.

- [ ] **Step 1: Add a highlight animation for citation jump-to**

In `frontend/app/globals.css`, add directly after the existing `.resolve { ... }` block (after line 144, before the `@media (prefers-reduced-motion: reduce)` block):

```css
/* --- citation jump-to: brief flash when a summary [n] link scrolls a quote
   card into view, so the reader can find which card it landed on --------- */

@keyframes cite-flash {
  from {
    background: var(--rule);
  }
  to {
    background: transparent;
  }
}

.cite-flash {
  animation: cite-flash 900ms ease-out;
}
```

- [ ] **Step 2: Update the page component**

Rewrite `frontend/app/ask/page.tsx` in full:

```tsx
"use client";

import Link from "next/link";
import { useState } from "react";
import { qaApi, type QuoteHit } from "@/lib/qaApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";

/*
  Extractive Q&A: every quote result is verbatim and already-grounded -
  never generated prose. See docs/superpowers/specs/
  2026-08-26-corpus-qa-design.md for why (the codebase has no generative
  LLM anywhere else, by deliberate "never invent" design).

  The optional summary panel below is the one exception: a local Ollama
  model may synthesize a short, cited rephrasing of the quotes already
  shown - never a replacement for them. See docs/superpowers/specs/
  2026-08-26-ollama-summary-layer-design.md for the grounding guarantees
  (citation-existence validation, fail-closed on an invalid citation).
*/

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [hits, setHits] = useState<QuoteHit[] | null>(null);
  const [summarizationAvailable, setSummarizationAvailable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [summary, setSummary] = useState<string | null>(null);
  const [citations, setCitations] = useState<number[]>([]);
  const [summarizing, setSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    setHits(null);
    setSummary(null);
    setSummaryError(null);
    try {
      const response = await qaApi.ask(text);
      setHits(response.hits);
      setSummarizationAvailable(response.summarization_available);
    } catch {
      setError("Couldn't reach the API. Is it running on port 8000?");
      setHits(null);
    } finally {
      setBusy(false);
    }
  }

  async function requestSummary() {
    if (!hits || hits.length === 0) return;
    setSummarizing(true);
    setSummaryError(null);
    try {
      const response = await qaApi.summarize(question.trim(), hits);
      setSummary(response.summary);
      setCitations(response.citations);
    } catch {
      setSummaryError("local LLM unavailable — quotes above are unaffected");
      setSummary(null);
    } finally {
      setSummarizing(false);
    }
  }

  function jumpToQuote(n: number) {
    const el = document.getElementById(`quote-${n}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("cite-flash");
    // force reflow so the animation restarts if the same card was just flashed
    void el.offsetWidth;
    el.classList.add("cite-flash");
  }

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <Nav />

      <section className="pt-12">
        <h1 className="display max-w-[24ch] text-[clamp(1.75rem,4vw,2.5rem)]">
          Ask a question, get grounded quotes back.
        </h1>
        <p className="mt-4 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every result is a real, already-extracted passage from a paper in the corpus — not a
          generated answer. Nothing here is invented; a quote either exists or it doesn&apos;t show
          up.
        </p>
        <p className="mt-3 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          This is different from the idea assessment on the home page: that produces a full
          synthesized report across many findings, while this page just answers one question at a
          time by finding the closest matching quotes already sitting in the corpus.
        </p>

        <form onSubmit={submit} className="mt-8 flex flex-wrap items-end gap-3">
          <label htmlFor="question" className="sr-only">
            question
          </label>
          <InfoTooltip
            label="How does this search work?"
            text="Your question first finds the closest papers by embedding similarity — the same search the corpus explorer uses — then re-ranks those papers' already-extracted claims and evidence against the question, and returns the best-matching passages as direct quotes."
          />
          <input
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={busy}
            placeholder="e.g. what are the limitations of graph transformers for fraud detection?"
            className="min-w-[20rem] flex-1 border-b-2 border-[var(--ink)] bg-transparent py-2 font-[family-name:var(--type-text)] text-[1.0625rem] placeholder:text-[var(--ink-faint)] focus:border-[var(--live)] focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!question.trim() || busy}
            className="eyebrow rounded-[2px] border border-[var(--ink)] px-4 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:border-[var(--rule)] disabled:text-[var(--ink-faint)] disabled:hover:bg-transparent disabled:hover:text-[var(--ink-faint)]"
          >
            {busy ? "searching…" : "ask"}
          </button>
        </form>

        {error && (
          <p className="mt-8 max-w-[58ch] border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
            {error}
          </p>
        )}

        {hits && hits.length === 0 && !error && (
          <p className="mt-10 text-[0.9375rem] text-[var(--ink-soft)]">
            No grounded evidence found for this question yet — try rephrasing, or the corpus may
            not have extracted claims covering this topic.
          </p>
        )}

        {hits && hits.length > 0 && summarizationAvailable && !summary && !summarizing && (
          <button
            type="button"
            onClick={requestSummary}
            className="eyebrow mt-10 rounded-[2px] border border-[var(--ink)] px-4 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)]"
          >
            ✨ synthesize a summary from these quotes
          </button>
        )}

        {summarizing && (
          <p className="mt-10 text-[0.9375rem] text-[var(--ink-soft)]">synthesizing…</p>
        )}

        {summaryError && (
          <p className="mt-10 max-w-[58ch] border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
            {summaryError}
          </p>
        )}

        {summary && hits && (
          <div className="mt-10 max-w-[68ch] border-l-2 border-[var(--near)] pl-4">
            <span className="eyebrow text-[var(--ink-faint)]">
              AI-synthesized from the quotes below — not independently verified
            </span>
            <p className="mt-2 font-[family-name:var(--type-text)] text-[1.0625rem] leading-relaxed text-[var(--ink)]">
              {renderSummaryWithCitationLinks(summary, jumpToQuote)}
            </p>
          </div>
        )}

        {hits && hits.length > 0 && (
          <ul className="mt-10">
            {hits.map((hit, i) => (
              <QuoteCard key={`${hit.paper_id}-${i}`} hit={hit} index={i} />
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

function renderSummaryWithCitationLinks(summary: string, onJump: (n: number) => void) {
  const parts = summary.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = Number(match[1]);
    return (
      <button
        key={i}
        type="button"
        onClick={() => onJump(n)}
        className="text-[var(--near)] underline underline-offset-2 hover:text-[var(--ink)]"
      >
        {part}
      </button>
    );
  });
}

function QuoteCard({ hit, index }: { hit: QuoteHit; index: number }) {
  return (
    <li
      id={`quote-${index + 1}`}
      className="resolve border-t border-[var(--rule-soft)] py-6 first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="eyebrow">{hit.claim_type.replace(/_/g, " ")}</span>
        {hit.section && (
          <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">{hit.section}</span>
        )}
      </div>

      <p className="mt-2 max-w-[68ch] font-[family-name:var(--type-text)] text-[1.0625rem] leading-relaxed text-[var(--ink)]">
        “{hit.text}”
      </p>

      <Link
        href={`/papers/${hit.paper_id}`}
        className="mt-2 inline-block text-[0.8125rem] text-[var(--ink-soft)] underline underline-offset-4 hover:text-[var(--ink)]"
      >
        {hit.paper_title} · {hit.paper_source}
      </Link>
    </li>
  );
}
```

Note the citation numbering: `[n]` markers from the backend are 1-indexed into the `hits` array the client sent, so `QuoteCard`'s `id` is `quote-${index + 1}` to match.

- [ ] **Step 3: Verify the frontend typechecks and builds**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: no type errors, build succeeds

- [ ] **Step 4: Manual browser verification**

This repo has no frontend test infra (existing convention — see the corpus Q&A spec's own Testing section). Verify live instead:

1. Set `OLLAMA_ENABLED=false` (or leave unset) in `.env`, restart the backend. Load `/ask`, ask a question with results. Confirm no "synthesize a summary" button appears — quotes render exactly as before.
2. Run `ollama pull qwen2.5:3b` if not already present (it is, per the earlier `ollama list` check — skip if so), confirm `ollama serve` is running. Set `OLLAMA_ENABLED=true` in `.env`, restart the backend.
3. Reload `/ask`, ask the same question. Confirm the "✨ synthesize a summary" button now appears above the quote list.
4. Click it. Confirm a busy state shows, then a summary panel appears above the quote list with the "not independently verified" label, `[n]` markers rendered as clickable links, and the quote list unchanged below it.
5. Click a `[n]` link. Confirm the page scrolls to the matching quote card and it briefly flashes.
6. Stop the Ollama server (or set `OLLAMA_TIMEOUT_SECONDS=1` with it still running slow) and repeat step 4. Confirm the inline "local LLM unavailable" message appears and the quote list is untouched.
7. Take a screenshot of the summary panel + quote list together for the record.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/ask/page.tsx frontend/app/globals.css
git commit -m "feat: add optional summary panel with citation links to the ask page"
```
