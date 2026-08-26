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
