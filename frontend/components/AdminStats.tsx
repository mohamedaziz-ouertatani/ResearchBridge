"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats } from "@/lib/api";
import { assessmentApi, type AssessmentSummary } from "@/lib/assessmentApi";
import type { PipelineRun, PipelineStatus } from "@/lib/adminApi";
import { YearStrip } from "@/components/YearStrip";

/*
  The stats tab: the same instrument-faceplate monochrome bars as the rest
  of the app (YearStrip, .tickstrip) rather than a colorful analytics-
  dashboard look or a charting library - the teal --near/--far ramp stays
  reserved for cosine distance (see globals.css), so every bar here is
  plain ink/rule grayscale.

  Deliberately reuses data already fetched elsewhere rather than adding new
  backend endpoints: /api/stats (corpus shape), the pipeline status this
  tab's parent already has (source/coverage/ingestion history), and
  /api/assessments (novelty breakdown + activity) - all existed before this
  tab, just not visualized together.
*/

const NOVELTY_ORDER = ["high", "medium", "low", "not_assessed"] as const;

export function AdminStats({ status }: { status: PipelineStatus }) {
  const [corpusStats, setCorpusStats] = useState<CorpusStats | null>(null);
  const [assessments, setAssessments] = useState<AssessmentSummary[] | null>(null);
  const [activeYear, setActiveYear] = useState<number | null>(null);

  useEffect(() => {
    assessmentApi
      .list("all")
      .then((page) => setAssessments(page.items))
      .catch(() => setAssessments(null));
  }, []);

  // Re-fetches on every year toggle rather than filtering client-side: the
  // category breakdown for a given year isn't part of the unfiltered
  // payload (only the top 20 categories corpus-wide are), so scoping it
  // to one year is a different query, not a client-side slice of this one.
  useEffect(() => {
    api
      .stats(activeYear !== null ? { year: activeYear } : undefined)
      .then(setCorpusStats)
      .catch(() => setCorpusStats(null));
  }, [activeYear]);

  return (
    <div className="space-y-10">
      <StatGroup title="corpus shape">
        {corpusStats && Object.keys(corpusStats.papers_by_year).length > 0 && (
          <div>
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">papers by year</span>
            <div className="mt-2">
              <YearStrip byYear={corpusStats.papers_by_year} activeYear={activeYear} onSelectYear={setActiveYear} />
            </div>
            {activeYear !== null && (
              <p className="mt-2 text-[0.75rem] text-[var(--near)]">
                Scoped to {activeYear} — categories, source breakdown, and coverage below all reflect this year
                only.{" "}
                <button
                  type="button"
                  onClick={() => setActiveYear(null)}
                  className="underline underline-offset-2 hover:text-[var(--ink)]"
                >
                  clear
                </button>
              </p>
            )}
          </div>
        )}
        {corpusStats && (
          <div className="mt-6">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">top categories</span>
            <div className="mt-2">
              <BarList counts={corpusStats.papers_by_category} limit={10} />
            </div>
          </div>
        )}
      </StatGroup>

      <StatGroup title="source & coverage">
        {corpusStats && (
          <>
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">papers by source</span>
            <div className="mt-2">
              <BarList counts={corpusStats.papers_by_source} />
            </div>

            <div className="mt-6 space-y-3">
              <ProportionBar
                label="with extracted claims"
                value={corpusStats.papers_with_claims}
                total={corpusStats.total_papers}
              />
              <ProportionBar
                label="with embeddings"
                value={corpusStats.embedded_papers}
                total={corpusStats.total_papers}
              />
            </div>
          </>
        )}
      </StatGroup>

      <StatGroup title="ingestion volume over time">
        <p className="mb-4 max-w-[58ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
          Records inserted per run, oldest to newest - limited to the run history already loaded above
          (not a full historical query). Not scoped by the year filter above: a run&apos;s timing has no fixed
          relationship to the publication year of the papers it fetched.
        </p>
        <div className="space-y-6">
          <IngestionVolume title="arXiv" runs={status.ingestion_runs.filter((r) => r.source === "arxiv")} />
          <IngestionVolume
            title="Springer Nature"
            runs={status.ingestion_runs.filter((r) => r.source === "springer")}
          />
          <IngestionVolume
            title="Semantic Scholar"
            runs={status.ingestion_runs.filter((r) => r.source === "semantic_scholar")}
          />
        </div>
      </StatGroup>

      <StatGroup title="assessment activity">
        {assessments && assessments.length > 0 ? (
          <>
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">created per day (most recent 50)</span>
            <div className="mt-2">
              <ActivitySparkline assessments={assessments} />
            </div>

            <div className="mt-6">
              <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">novelty level</span>
              <div className="mt-2">
                <BarList
                  counts={Object.fromEntries(
                    NOVELTY_ORDER.map((level) => [
                      level.replace("_", " "),
                      assessments.filter((a) => a.novelty_level === level).length,
                    ]),
                  )}
                  preserveOrder
                />
              </div>
            </div>
          </>
        ) : (
          <p className="text-[0.875rem] text-[var(--ink-faint)]">No assessments yet.</p>
        )}
      </StatGroup>
    </div>
  );
}

function StatGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-[var(--rule-soft)] pb-10 last:border-b-0">
      <span className="eyebrow">{title}</span>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** A ranked horizontal bar per entry, monochrome, sorted desc unless preserveOrder. */
function BarList({
  counts,
  limit,
  preserveOrder = false,
}: {
  counts: Record<string, number>;
  limit?: number;
  preserveOrder?: boolean;
}) {
  let entries = Object.entries(counts);
  if (!preserveOrder) entries = entries.sort((a, b) => b[1] - a[1]);
  if (limit) entries = entries.slice(0, limit);

  const peak = Math.max(...entries.map(([, count]) => count), 1);

  if (entries.length === 0) return <p className="text-[0.875rem] text-[var(--ink-faint)]">No data.</p>;

  return (
    <ul className="space-y-1.5">
      {entries.map(([label, count]) => (
        <li key={label} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3">
          <span className="truncate text-[0.8125rem] text-[var(--ink-soft)]" title={label}>
            {label}
          </span>
          <span className="h-2 rounded-[1px] bg-[var(--rule-soft)]">
            <span
              className="block h-full rounded-[1px] bg-[var(--ink-faint)]"
              style={{ width: `${Math.max((count / peak) * 100, 2)}%` }}
            />
          </span>
          <span className="readout text-right text-[0.75rem] text-[var(--ink-faint)] tabular-nums">
            {count.toLocaleString()}
          </span>
        </li>
      ))}
    </ul>
  );
}

function ProportionBar({ label, value, total }: { label: string; value: number; total: number }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-[0.8125rem] text-[var(--ink-soft)]">{label}</span>
        <span className="readout text-[0.75rem] text-[var(--ink-faint)] tabular-nums">
          {value.toLocaleString()} / {total.toLocaleString()}
        </span>
      </div>
      <span className="mt-1 block h-2 rounded-[1px] bg-[var(--rule-soft)]">
        <span className="block h-full rounded-[1px] bg-[var(--ink-faint)]" style={{ width: `${pct}%` }} />
      </span>
    </div>
  );
}

/** Bar-per-run strip, oldest to newest, reusing the .tickstrip pattern. */
function IngestionVolume({ title, runs }: { title: string; runs: PipelineRun[] }) {
  const ordered = [...runs].sort((a, b) => a.started_at.localeCompare(b.started_at));
  const peak = Math.max(...ordered.map((r) => r.counts.records_inserted ?? 0), 1);

  return (
    <div>
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{title}</span>
      {ordered.length === 0 ? (
        <p className="mt-2 text-[0.8125rem] text-[var(--ink-faint)]">No runs recorded yet.</p>
      ) : (
        <div className="tickstrip mt-2 h-10" style={{ ["--cols" as string]: ordered.length }}>
          {ordered.map((run) => {
            const inserted = run.counts.records_inserted ?? 0;
            return (
              <div
                key={run.id}
                className="tick"
                title={`${new Date(run.started_at).toLocaleDateString()} — ${inserted.toLocaleString()} inserted`}
              >
                <span
                  className="tick-bar"
                  style={{ height: `${Math.max((inserted / peak) * 100, inserted > 0 ? 4 : 0)}%` }}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/** Day-bucketed count of assessment creation, oldest to newest. */
function ActivitySparkline({ assessments }: { assessments: AssessmentSummary[] }) {
  const byDay = new Map<string, number>();
  for (const a of assessments) {
    const day = new Date(a.created_at).toDateString();
    byDay.set(day, (byDay.get(day) ?? 0) + 1);
  }
  const days = [...byDay.keys()].sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
  const peak = Math.max(...byDay.values(), 1);

  return (
    <div className="tickstrip h-10" style={{ ["--cols" as string]: days.length }}>
      {days.map((day) => {
        const count = byDay.get(day) ?? 0;
        return (
          <div key={day} className="tick" title={`${day} — ${count} assessment${count === 1 ? "" : "s"}`}>
            <span className="tick-bar" style={{ height: `${Math.max((count / peak) * 100, 4)}%` }} />
          </div>
        );
      })}
    </div>
  );
}
