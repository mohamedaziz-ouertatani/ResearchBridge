"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type CorpusStats, type PaperSummary, type SearchHit } from "@/lib/api";
import { GaugeLegend } from "@/components/ProximityGauge";
import { PaperRow } from "@/components/PaperRow";
import { YearStrip } from "@/components/YearStrip";

type Mode = "browse" | "search";

export default function Explorer() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("browse");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [year, setYear] = useState<number | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [showExcluded, setShowExcluded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.stats().then(setStats).catch(() => setError("Can't reach the API. Is it running on port 8000?"));
  }, []);

  const loadBrowse = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const page = await api.papers({
        limit: 25,
        year: year ?? undefined,
        category: category ?? undefined,
        include_excluded: showExcluded || undefined,
      });
      setPapers(page.items);
      setTotal(page.total);
    } catch {
      setError("Couldn't load papers.");
    } finally {
      setBusy(false);
    }
  }, [year, category, showExcluded]);

  useEffect(() => {
    if (mode === "browse") void loadBrowse();
  }, [mode, loadBrowse]);

  async function runSearch(event: React.FormEvent) {
    event.preventDefault();
    const q = query.trim();
    if (!q) return;

    setBusy(true);
    setError(null);
    setMode("search");
    try {
      setHits(await api.search(q, 12));
    } catch {
      setError("Search failed. The embedding model may still be loading — try again.");
    } finally {
      setBusy(false);
    }
  }

  function clearSearch() {
    setQuery("");
    setHits([]);
    setMode("browse");
    inputRef.current?.focus();
  }

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
        <div className="flex items-baseline gap-5">
          <Link href="/annotate" className="eyebrow hover:text-[var(--ink)]">
            annotation workbench →
          </Link>
          <Link href="/gaps" className="eyebrow hover:text-[var(--ink)]">
            gap review →
          </Link>
          <span className="eyebrow">
            {stats ? `${stats.total_papers.toLocaleString()} papers · ${stats.total_authors.toLocaleString()} authors` : "connecting"}
          </span>
        </div>
      </header>

      {/* Hero: the aperture. Type meaning in, the corpus resolves by distance. */}
      <section className="pt-16 pb-10">
        <span className="eyebrow">supporting infrastructure</span>
        <h1 className="display mt-3 max-w-[16ch] text-[clamp(2.25rem,5.5vw,3.5rem)]">
          Search the corpus by meaning.
        </h1>
        <p className="mt-4 max-w-[52ch] text-[1.0625rem] leading-relaxed text-[var(--ink-soft)]">
          The literature an assessment draws on. Every result is scored by how close it sits to
          your query in embedding space — not by which words it happens to contain.
        </p>

        <form onSubmit={runSearch} className="mt-8">
          <div className="flex items-center gap-3 border-b-2 border-[var(--ink)] pb-2 focus-within:border-[var(--live)]">
            <span aria-hidden className="readout text-[var(--ink-faint)]">
              ▸
            </span>
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="describe what you're looking for"
              aria-label="Search the corpus by meaning"
              className="w-full bg-transparent py-1.5 font-[family-name:var(--type-display)] text-[1.0625rem] placeholder:text-[var(--ink-faint)] focus:outline-none"
            />
            {query && (
              <button
                type="button"
                onClick={clearSearch}
                className="eyebrow shrink-0 hover:text-[var(--ink)]"
              >
                clear
              </button>
            )}
            <button
              type="submit"
              disabled={!query.trim() || busy}
              className="eyebrow shrink-0 text-[var(--ink)] disabled:text-[var(--ink-faint)]"
            >
              {busy && mode === "search" ? "measuring" : "search"}
            </button>
          </div>
        </form>
      </section>

      {/* The corpus's own shape, doubling as a year filter. */}
      {stats && mode === "browse" && (
        <section className="border-t border-[var(--rule)] pt-6">
          <div className="mb-4 flex items-baseline justify-between gap-4">
            <span className="eyebrow">corpus by year</span>
            {year && (
              <button onClick={() => setYear(null)} className="eyebrow hover:text-[var(--ink)]">
                clear {year} ✕
              </button>
            )}
          </div>
          <YearStrip byYear={stats.papers_by_year} activeYear={year} onSelectYear={setYear} />

          <div className="mt-8 flex flex-wrap items-center gap-2">
            <span className="eyebrow mr-1">categories</span>
            {Object.entries(stats.papers_by_category)
              .slice(0, 8)
              .map(([name, count]) => {
                const isActive = category === name;
                return (
                  <button
                    key={name}
                    onClick={() => setCategory(isActive ? null : name)}
                    aria-pressed={isActive}
                    className={`readout rounded-[2px] border px-2 py-1 text-[0.6875rem] transition-colors ${
                      isActive
                        ? "border-[var(--near)] bg-[var(--near)] text-white"
                        : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)]"
                    }`}
                  >
                    {name} <span className="opacity-60">{count}</span>
                  </button>
                );
              })}
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button
              onClick={() => setShowExcluded((v) => !v)}
              aria-pressed={showExcluded}
              className={`readout rounded-[2px] border px-2 py-1 text-[0.6875rem] transition-colors ${
                showExcluded
                  ? "border-[var(--live)] text-[var(--live)]"
                  : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)]"
              }`}
            >
              show excluded
            </button>
          </div>
        </section>
      )}

      {error && (
        <p className="mt-10 border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] text-[var(--ink-soft)]">
          {error}
        </p>
      )}

      {/* Results */}
      <section className="mt-12">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] pb-3">
          <span className="eyebrow">
            {mode === "search"
              ? `${hits.length} nearest to “${query.trim()}”`
              : `${total.toLocaleString()} papers${year ? ` · ${year}` : ""}${category ? ` · ${category}` : ""}`}
          </span>
          {mode === "search" && hits.length > 0 && <GaugeLegend />}
        </div>

        {busy && (
          <p className="eyebrow py-10">{mode === "search" ? "measuring distances…" : "loading…"}</p>
        )}

        {!busy && mode === "search" && hits.length === 0 && !error && (
          <p className="py-10 text-[0.9375rem] text-[var(--ink-soft)]">
            Nothing came back. Try describing the idea rather than naming it.
          </p>
        )}

        {!busy && (
          <ul>
            {mode === "search"
              ? hits.map((hit, i) => (
                  <PaperRow key={hit.paper.id} paper={hit.paper} distance={hit.distance} index={i} />
                ))
              : papers.map((paper, i) => <PaperRow key={paper.id} paper={paper} index={i} />)}
          </ul>
        )}
      </section>
    </main>
  );
}
