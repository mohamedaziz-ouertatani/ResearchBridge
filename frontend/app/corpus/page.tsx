"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type CorpusStats, type PaperSummary, type SearchHit } from "@/lib/api";
import { GaugeLegend } from "@/components/ProximityGauge";
import { PaperRow } from "@/components/PaperRow";
import { YearStrip } from "@/components/YearStrip";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { SkeletonRows } from "@/components/Skeleton";

type Mode = "browse" | "search";
type Sort = "date_desc" | "date_asc" | "title_asc";

const SORT_OPTIONS: { value: Sort; label: string }[] = [
  { value: "date_desc", label: "newest" },
  { value: "date_asc", label: "oldest" },
  { value: "title_asc", label: "title A–Z" },
];

export default function Explorer() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("browse");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [year, setYear] = useState<number | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [sort, setSort] = useState<Sort>("date_desc");
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
        source: source ?? undefined,
        include_excluded: showExcluded || undefined,
        sort,
      });
      setPapers(page.items);
      setTotal(page.total);
    } catch {
      setError("Couldn't load papers.");
    } finally {
      setBusy(false);
    }
  }, [year, category, source, sort, showExcluded]);

  useEffect(() => {
    // loadBrowse resets busy/error before fetching, keyed on mode/year/category/
    // source/sort/showExcluded - intentional, not the accidental-derived-state
    // case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
      <Nav />

      {/* Hero: the aperture. Type meaning in, the corpus resolves by distance. */}
      <section className="pt-16 pb-10">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <span className="eyebrow">supporting infrastructure</span>
          <span className="eyebrow text-[var(--ink-faint)]">
            {stats ? `${stats.total_papers.toLocaleString()} papers · ${stats.total_authors.toLocaleString()} authors` : "connecting"}
          </span>
        </div>
        <h1 className="display mt-3 max-w-[16ch] text-[clamp(2.25rem,5.5vw,3.5rem)]">
          Search the corpus by meaning.
        </h1>
        <p className="mt-4 max-w-[52ch] text-[1.0625rem] leading-relaxed text-[var(--ink-soft)]">
          The literature an assessment draws on. Every result is scored by how close it sits to
          your query in embedding space — not by which words it happens to contain, so a search
          for &quot;detecting fraud in real time&quot; can surface a paper that never uses the
          word &quot;fraud&quot; at all.
        </p>
        <p className="mt-3 max-w-[52ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          This page browses the same collection of papers that idea assessments and gap detection
          are built on — it doesn&apos;t assess anything itself, it just lets you look at what&apos;s
          in there and how it was gathered.
        </p>

        <form onSubmit={runSearch} className="mt-8">
          <div className="flex items-center gap-3 border-b-2 border-[var(--ink)] pb-2 focus-within:border-[var(--live)]">
            <span aria-hidden className="readout text-[var(--ink-faint)]">
              ▸
            </span>
            <InfoTooltip
              label="What does search do?"
              text="Searches by meaning, not keywords: your query is embedded and compared against every paper's embedding, then ranked by how close they sit (the teal/gray gauge on each result). Clear the box to go back to browsing by year and category."
            />
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
            <span className="eyebrow inline-flex items-center gap-1.5">
              corpus by year
              <InfoTooltip text="Each bar is how many papers were published that year. Click a bar to filter the list below to just that year; click it again (or the clear button) to remove the filter." />
            </span>
            {year && (
              <button onClick={() => setYear(null)} className="eyebrow hover:text-[var(--ink)]">
                clear {year} ✕
              </button>
            )}
          </div>
          <YearStrip byYear={stats.papers_by_year} activeYear={year} onSelectYear={setYear} />

          <div className="mt-8 flex flex-wrap items-center gap-2">
            <span className="eyebrow mr-1 inline-flex items-center gap-1.5">
              categories
              <InfoTooltip text="Each source paper is tagged with a subject category on ingestion. Click one to filter to just that category; click it again to clear the filter. Only the 8 largest categories are shown here." />
            </span>
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

          <div className="mt-8 flex flex-wrap items-center gap-2">
            <span className="eyebrow mr-1 inline-flex items-center gap-1.5">
              sources
              <InfoTooltip text="Where each paper was ingested from — e.g. arXiv or Springer Nature. Click one to filter to just that source; click it again to clear the filter." />
            </span>
            {Object.entries(stats.papers_by_source).map(([name, count]) => {
              const isActive = source === name;
              return (
                <button
                  key={name}
                  onClick={() => setSource(isActive ? null : name)}
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

          <div className="mt-8 flex flex-wrap items-center gap-2">
            <span className="eyebrow mr-1 inline-flex items-center gap-1.5">
              sort by
              <InfoTooltip text="Order the list below. Applies only while browsing — search results are always ranked by how closely they match your query." />
            </span>
            {SORT_OPTIONS.map(({ value, label }) => {
              const isActive = sort === value;
              return (
                <button
                  key={value}
                  onClick={() => setSort(value)}
                  aria-pressed={isActive}
                  className={`readout rounded-[2px] border px-2 py-1 text-[0.6875rem] transition-colors ${
                    isActive
                      ? "border-[var(--near)] bg-[var(--near)] text-white"
                      : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)]"
                  }`}
                >
                  {label}
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
            <InfoTooltip text="Some papers are marked excluded — e.g. withdrawn, or found unsuitable during review — and are hidden by default so they don't skew search and browse. Toggle this to include them anyway." />
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
              : `${total.toLocaleString()} papers${year ? ` · ${year}` : ""}${category ? ` · ${category}` : ""}${source ? ` · ${source}` : ""}${showExcluded ? " · including excluded" : ""}`}
          </span>
          {mode === "search" && hits.length > 0 && <GaugeLegend />}
        </div>

        {busy && (
          <>
            <p className="eyebrow pt-2 pb-4">{mode === "search" ? "measuring distances…" : "loading…"}</p>
            <SkeletonRows count={5} />
          </>
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
