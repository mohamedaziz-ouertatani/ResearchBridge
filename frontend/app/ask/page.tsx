"use client";

import Link from "next/link";
import { useState } from "react";
import { qaApi, type QuoteHit } from "@/lib/qaApi";

/*
  Extractive Q&A: every result is a verbatim, already-grounded quote from
  the corpus - never generated prose. See docs/superpowers/specs/
  2026-08-26-corpus-qa-design.md for why (the codebase has no generative
  LLM anywhere, by deliberate "never invent" design).
*/

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [hits, setHits] = useState<QuoteHit[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    try {
      const response = await qaApi.ask(text);
      setHits(response.hits);
    } catch {
      setError("Couldn't reach the API. Is it running on port 8000?");
      setHits(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
        <span className="eyebrow">ask the corpus</span>
      </header>

      <section className="pt-12">
        <h1 className="display max-w-[24ch] text-[clamp(1.75rem,4vw,2.5rem)]">
          Ask a question, get grounded quotes back.
        </h1>
        <p className="mt-4 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every result is a real, already-extracted passage from a paper in the corpus — not a
          generated answer. Nothing here is invented; a quote either exists or it doesn&apos;t show
          up.
        </p>

        <form onSubmit={submit} className="mt-8 flex flex-wrap items-end gap-3">
          <label htmlFor="question" className="sr-only">
            question
          </label>
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

function QuoteCard({ hit, index }: { hit: QuoteHit; index: number }) {
  return (
    <li
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
