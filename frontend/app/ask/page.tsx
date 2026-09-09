"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  qaApi,
  type QaCollection,
  type QaCollectionSummary,
  type QuoteHit,
} from "@/lib/qaApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { EvidenceReviewControl } from "@/components/EvidenceReviewControl";

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
  const [collections, setCollections] = useState<QaCollectionSummary[]>([]);
  const [selectedCollectionId, setSelectedCollectionId] = useState<
    string | null
  >(null);
  const [collection, setCollection] = useState<QaCollection | null>(null);
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(
    null,
  );
  const [newCollectionTitle, setNewCollectionTitle] = useState("");
  const [renameTitle, setRenameTitle] = useState("");
  const [collectionBusy, setCollectionBusy] = useState(false);
  const [collectionLoading, setCollectionLoading] = useState(true);
  const [collectionError, setCollectionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [summarizing, setSummarizing] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const selectedQuestion =
    collection?.questions.find((item) => item.id === selectedQuestionId) ??
    null;
  const hits = selectedQuestion?.hits ?? null;
  const summary = selectedQuestion?.summary ?? null;

  useEffect(() => {
    let cancelled = false;
    // Loading the user's local collection list on page entry is intentional.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCollectionLoading(true);
    setCollectionError(null);
    qaApi
      .listCollections()
      .then(async (items) => {
        if (cancelled) return;
        if (items.length === 0) {
          const created = await qaApi.createCollection("Research notes");
          if (cancelled) return;
          items = [created];
        }
        setCollections(items);
        setSelectedCollectionId((current) =>
          current && items.some((item) => item.id === current)
            ? current
            : items[0].id,
        );
      })
      .catch(() => {
        if (!cancelled)
          setCollectionError(
            "Couldn't load Q&A collections. Is the API running on port 8000?",
          );
      })
      .finally(() => {
        if (!cancelled) setCollectionLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedCollectionId) {
      // Clear the previous collection while the user has no active selection.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setCollection(null);
      return;
    }
    let cancelled = false;
    setCollectionError(null);
    qaApi
      .getCollection(selectedCollectionId)
      .then((loaded) => {
        if (cancelled) return;
        setCollection(loaded);
        setRenameTitle(loaded.title);
      })
      .catch(() => {
        if (!cancelled)
          setCollectionError("Couldn't load that Q&A collection.");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedCollectionId]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    setSummaryError(null);
    if (!selectedCollectionId) {
      setError("Create or select a collection before asking a question.");
      setBusy(false);
      return;
    }
    try {
      const saved = await qaApi.askInCollection(selectedCollectionId, text);
      setCollection((current) =>
        current
          ? { ...current, questions: [saved, ...current.questions] }
          : current,
      );
      setCollections((current) =>
        current.map((item) =>
          item.id === selectedCollectionId
            ? { ...item, question_count: item.question_count + 1 }
            : item,
        ),
      );
      setSelectedQuestionId(saved.id);
      setQuestion("");
    } catch {
      setError("Couldn't save the question. Is the API running on port 8000?");
    } finally {
      setBusy(false);
    }
  }

  async function requestSummary() {
    if (!selectedQuestion || selectedQuestion.hits.length === 0) return;
    setSummarizing(true);
    setSummaryError(null);
    try {
      const updated = await qaApi.summarizeQuestion(selectedQuestion.id);
      setCollection((current) =>
        current
          ? {
              ...current,
              questions: current.questions.map((item) =>
                item.id === updated.id ? updated : item,
              ),
            }
          : current,
      );
    } catch {
      setSummaryError("local LLM unavailable — quotes above are unaffected");
    } finally {
      setSummarizing(false);
    }
  }

  async function createCollection() {
    const title = newCollectionTitle.trim();
    if (!title) return;
    setCollectionBusy(true);
    setCollectionError(null);
    try {
      const created = await qaApi.createCollection(title);
      setCollections((current) => [created, ...current]);
      setSelectedCollectionId(created.id);
      setNewCollectionTitle("");
    } catch {
      setCollectionError("Couldn't create that collection.");
    } finally {
      setCollectionBusy(false);
    }
  }

  async function renameCollection(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedCollectionId || !renameTitle.trim()) return;
    setCollectionBusy(true);
    try {
      const renamed = await qaApi.renameCollection(
        selectedCollectionId,
        renameTitle.trim(),
      );
      setCollections((current) =>
        current.map((item) => (item.id === renamed.id ? renamed : item)),
      );
      setCollection((current) =>
        current
          ? { ...current, title: renamed.title, updated_at: renamed.updated_at }
          : current,
      );
    } catch {
      setCollectionError("Couldn't rename that collection.");
    } finally {
      setCollectionBusy(false);
    }
  }

  async function deleteCollection() {
    if (
      !selectedCollectionId ||
      !window.confirm("Delete this collection and its saved questions?")
    )
      return;
    setCollectionBusy(true);
    try {
      await qaApi.deleteCollection(selectedCollectionId);
      const remaining = collections.filter(
        (item) => item.id !== selectedCollectionId,
      );
      setCollections(remaining);
      setSelectedCollectionId(remaining[0]?.id ?? null);
      setCollection(null);
      setSelectedQuestionId(null);
    } catch {
      setCollectionError("Couldn't delete that collection.");
    } finally {
      setCollectionBusy(false);
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
          Every result is a real, already-extracted passage from a paper in the
          corpus — not a generated answer. Nothing here is invented; a quote
          either exists or it doesn&apos;t show up.
        </p>
        <p className="mt-3 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          This is different from the idea assessment on the home page: that
          produces a full synthesized report across many findings, while this
          page just answers one question at a time by finding the closest
          matching quotes already sitting in the corpus.
        </p>

        <section className="mt-8 border-y border-[var(--rule)] py-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex min-w-[16rem] flex-1 flex-col gap-1">
              <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                collection
              </span>
              <select
                value={selectedCollectionId ?? ""}
                onChange={(event) => {
                  setSelectedCollectionId(event.target.value || null);
                  setSelectedQuestionId(null);
                  setQuestion("");
                }}
                disabled={collectionLoading || collectionBusy}
                className="border-b border-[var(--ink)] bg-transparent py-2 text-[0.9375rem] focus:outline-none disabled:opacity-50"
              >
                {collections.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title} ({item.question_count})
                  </option>
                ))}
              </select>
            </label>
            <form onSubmit={renameCollection} className="flex items-end gap-2">
              <label className="flex flex-col gap-1">
                <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                  rename
                </span>
                <input
                  value={renameTitle}
                  onChange={(event) => setRenameTitle(event.target.value)}
                  maxLength={120}
                  disabled={!collection || collectionBusy}
                  className="w-[12rem] border-b border-[var(--rule)] bg-transparent py-2 text-[0.8125rem] focus:border-[var(--ink)] focus:outline-none disabled:opacity-50"
                />
              </label>
              <button
                type="submit"
                disabled={!collection || !renameTitle.trim() || collectionBusy}
                className="eyebrow py-2 text-[var(--ink-soft)] hover:text-[var(--ink)] disabled:opacity-40"
              >
                save name
              </button>
            </form>
            <button
              type="button"
              onClick={deleteCollection}
              disabled={!collection || collectionBusy}
              className="eyebrow py-2 text-[var(--ink-faint)] hover:text-[var(--live)] disabled:opacity-40"
            >
              delete collection
            </button>
          </div>

          <form
            onSubmit={(event) => {
              event.preventDefault();
              void createCollection();
            }}
            className="mt-4 flex max-w-[28rem] items-end gap-2"
          >
            <label className="flex min-w-0 flex-1 flex-col gap-1">
              <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                new collection
              </span>
              <input
                value={newCollectionTitle}
                onChange={(event) => setNewCollectionTitle(event.target.value)}
                placeholder="e.g. federated learning review"
                maxLength={120}
                className="border-b border-[var(--rule)] bg-transparent py-2 text-[0.8125rem] focus:border-[var(--ink)] focus:outline-none"
              />
            </label>
            <button
              type="submit"
              disabled={!newCollectionTitle.trim() || collectionBusy}
              className="eyebrow py-2 text-[var(--ink-soft)] hover:text-[var(--ink)] disabled:opacity-40"
            >
              create
            </button>
          </form>
        </section>

        {collectionError && (
          <p className="mt-5 max-w-[58ch] border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
            {collectionError}
          </p>
        )}

        {collection && collection.questions.length > 0 && (
          <aside className="mt-8 border-l border-[var(--rule)] pl-4">
            <span className="eyebrow">saved questions</span>
            <ul className="mt-2 space-y-1">
              {collection.questions.map((saved) => (
                <li key={saved.id}>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedQuestionId(saved.id);
                      setQuestion(saved.question);
                      setSummaryError(null);
                    }}
                    className={`block max-w-[64ch] text-left text-[0.8125rem] leading-relaxed hover:text-[var(--ink)] ${saved.id === selectedQuestionId ? "text-[var(--ink)]" : "text-[var(--ink-faint)]"}`}
                  >
                    {saved.question}
                  </button>
                </li>
              ))}
            </ul>
          </aside>
        )}

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
            disabled={!question.trim() || busy || !selectedCollectionId}
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
            No grounded evidence found for this question yet — try rephrasing,
            or the corpus may not have extracted claims covering this topic.
          </p>
        )}

        {hits && hits.length > 0 && !summary && !summarizing && (
          <button
            type="button"
            onClick={requestSummary}
            className="eyebrow mt-10 rounded-[2px] border border-[var(--ink)] px-4 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)]"
          >
            ✨ synthesize a summary from these quotes
          </button>
        )}

        {summarizing && (
          <p className="mt-10 text-[0.9375rem] text-[var(--ink-soft)]">
            synthesizing…
          </p>
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
              <QuoteCard
                key={`${hit.paper_id}-${i}`}
                hit={hit}
                index={i}
                questionId={selectedQuestionId}
              />
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

function renderSummaryWithCitationLinks(
  summary: string,
  onJump: (n: number) => void,
) {
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

function QuoteCard({
  hit,
  index,
  questionId,
}: {
  hit: QuoteHit;
  index: number;
  questionId: string | null;
}) {
  return (
    <li
      id={`quote-${index + 1}`}
      className="resolve border-t border-[var(--rule-soft)] py-6 first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="eyebrow">{hit.claim_type.replace(/_/g, " ")}</span>
        {hit.section && (
          <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
            {hit.section}
          </span>
        )}
      </div>

      <p className="mt-2 max-w-[68ch] font-[family-name:var(--type-text)] text-[1.0625rem] leading-relaxed text-[var(--ink)]">
        “{hit.text}”
      </p>

      {questionId && hit.evidence_id && (
        <EvidenceReviewControl
          evidenceId={hit.evidence_id}
          surfaceType="qa_question"
          surfaceId={questionId}
          role={hit.claim_type}
        />
      )}

      <Link
        href={`/papers/${hit.paper_id}`}
        className="mt-2 inline-block text-[0.8125rem] text-[var(--ink-soft)] underline underline-offset-4 hover:text-[var(--ink)]"
      >
        {hit.paper_title} · {hit.paper_source}
      </Link>
    </li>
  );
}
