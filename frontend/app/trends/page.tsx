"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats, type TrendsData } from "@/lib/api";
import { Nav } from "@/components/Nav";

// Fixed order matching the blueprint's documented claim_type list, not
// whatever order the API returns keys in - keeps the page from reflowing
// as you switch categories.
const CLAIM_TYPE_ORDER = [
  "problem",
  "method",
  "dataset",
  "metric",
  "result",
  "limitation",
  "research_gap",
  "application",
  "contribution",
];

const SELECT_CLASS =
  "eyebrow rounded-[2px] border border-[var(--rule)] bg-transparent px-2 py-1 text-[0.6875rem] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none";

export default function TrendsPage() {
  const [categories, setCategories] = useState<string[]>([]);
  const [category, setCategory] = useState<string>("");
  const [data, setData] = useState<TrendsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .stats()
      .then((stats: CorpusStats) => {
        const sorted = Object.keys(stats.papers_by_category).sort(
          (a, b) => stats.papers_by_category[b] - stats.papers_by_category[a],
        );
        setCategories(sorted);
        if (sorted.length > 0) setCategory(sorted[0]);
      })
      .catch(() => setError("Couldn't load categories."));
  }, []);

  useEffect(() => {
    if (!category) return;
    // Resetting data/error before a fetch keyed on `category` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    api
      .trends(category)
      .then(setData)
      .catch(() => setError("Couldn't load trends."));
  }, [category]);

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">trends</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          How many extracted claims of each type appear per year, within one category - a real
          count of what&apos;s already in the corpus, not an inferred pattern.
        </p>

        <label className="mt-6 flex items-center gap-2">
          <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">category</span>
          <select value={category} onChange={(e) => setCategory(e.target.value)} className={SELECT_CLASS}>
            {categories.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

        {!error && !data && <p className="eyebrow py-16">loading…</p>}

        {!error && data && data.years.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            No extracted claims yet for this category.
          </p>
        )}

        {!error && data && data.years.length > 0 && (
          <div className="mt-8 space-y-6">
            {CLAIM_TYPE_ORDER.filter((claimType) => data.series[claimType]).map((claimType) => (
              <TrendStrip
                key={claimType}
                label={claimType.replace("_", " ")}
                years={data.years}
                counts={data.series[claimType]}
              />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

function TrendStrip({ label, years, counts }: { label: string; years: number[]; counts: number[] }) {
  const peak = Math.max(...counts, 1);

  return (
    <div>
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{label}</span>
      <div className="tickstrip mt-2 h-10" style={{ ["--cols" as string]: years.length }}>
        {years.map((year, i) => (
          <div key={year} className="tick" title={`${year} — ${counts[i].toLocaleString()}`}>
            <span
              className="tick-bar"
              style={{ height: `${Math.max((counts[i] / peak) * 100, counts[i] > 0 ? 4 : 0)}%` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
