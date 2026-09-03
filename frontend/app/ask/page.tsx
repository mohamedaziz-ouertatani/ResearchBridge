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
  const [answeredQuestion, setAnsweredQuestion] = useState("");
  const [hits, setHits] = useState<QuoteHit[] | null>(null);
  const [summarizationAvailable, setSummarizationAvailable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [summary, setSummary] = useState<string | null>(null);
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
      setAnsweredQuestion(text);
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
      const response = await qaApi.summarize(answeredQuestion, hits);
      setSummary(response.summary);
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
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
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
