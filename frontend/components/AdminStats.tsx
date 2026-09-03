"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats } from "@/lib/api";
import { assessmentApi, type AssessmentSummary } from "@/lib/assessmentApi";
import type { GapReviewStats, PipelineRun, PipelineStatus } from "@/lib/adminApi";
import { InfoTooltip } from "@/components/InfoTooltip";
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

const GAP_RATING_DIMENSIONS: { key: keyof GapReviewStats & `mean_${string}`; label: string }[] = [
  { key: "mean_correctness", label: "correctness" },
  { key: "mean_relevance", label: "relevance" },
  { key: "mean_novelty", label: "novelty" },
  { key: "mean_evidence_support", label: "evidence support" },
  { key: "mean_usefulness", label: "usefulness" },
];

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

      <StatGroup title="corpus health">
        <p className="mb-4 max-w-[58ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
          Papers stuck mid-pipeline or unreachable by a citation source - not necessarily broken, just
          worth knowing about before trusting a downstream count.
        </p>
        <div className="grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-5">
          <HealthStat
            label="missing DOI"
            value={status.corpus_health.missing_doi}
            total={status.total_papers}
            info="Can't be reached by the CrossRef citation pass, which looks papers up by DOI."
          />
          <HealthStat
            label="excluded"
            value={status.corpus_health.excluded}
            total={status.total_papers}
            info="Marked excluded from the corpus via the paper detail page."
          />
          <HealthStat
            label="claims, no embeddings"
            value={status.corpus_health.claims_without_embeddings}
            total={status.total_papers}
            info="Extracted but not yet embedded - not searchable by meaning or eligible for gap detection yet."
          />
          <HealthStat
            label="no citation coverage"
            value={status.corpus_health.no_citation_coverage}
            total={status.total_papers}
            info="Eligible for Semantic Scholar or CrossRef citation fetching (has a DOI, or came from Semantic Scholar) but hasn't been fetched yet."
          />
          <HealthStat
            label="not gap processed"
            value={status.corpus_health.not_gap_processed}
            total={status.total_papers}
            info="Embedded and not excluded, but gap detection hasn't used it as a seed paper yet - the backlog the next gap detection run would work through."
          />
        </div>
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
          <IngestionVolume title="CORE" runs={status.ingestion_runs.filter((r) => r.source === "core")} />
        </div>

        {Object.keys(status.ingestion_errors_by_type).length > 0 && (
          <div className="mt-8">
            <span className="eyebrow inline-flex items-center gap-1.5 text-[0.625rem] text-[var(--ink-faint)]">
              recent errors by type
              <InfoTooltip text="Grouped from the most recent ingestion_errors rows across every source - a sample of what's failing lately, not an all-time count." />
            </span>
            <div className="mt-2">
              <BarList counts={status.ingestion_errors_by_type} />
            </div>
          </div>
        )}
      </StatGroup>

      <StatGroup title="gap review">
        <GapDetectionLastRun runs={status.gap_detection_runs} />

        <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-3">
          <Stat label="pending" value={status.gap_stats.pending} />
          <Stat label="approved" value={status.gap_stats.approved} />
          <Stat label="rejected" value={status.gap_stats.rejected} />
        </dl>

        {GAP_RATING_DIMENSIONS.some(({ key }) => status.gap_stats[key] !== null) && (
          <div className="mt-6">
            <span className="eyebrow inline-flex items-center gap-1.5 text-[0.625rem] text-[var(--ink-faint)]">
              mean rating (0–3), rated gaps only
              <InfoTooltip text="Sec 44's five human-evaluation dimensions, averaged across whatever gaps a reviewer has rated on each - not every reviewed gap needs a rating, so these can be based on a smaller count than the totals above." />
            </span>
            <div className="mt-2 grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-5">
              {GAP_RATING_DIMENSIONS.map(({ key, label }) => (
                <div key={key}>
                  <dt className="text-[0.75rem] text-[var(--ink-soft)]">{label}</dt>
                  <dd className="readout mt-1 text-[1.125rem] tabular-nums">
                    {status.gap_stats[key] !== null ? status.gap_stats[key]!.toFixed(2) : "—"}
                  </dd>
                </div>
              ))}
            </div>
          </div>
        )}
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

/** A single labeled count, no total or tooltip - used by the gap review
    group, where each number stands alone rather than being a share of
    something. */
function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{label}</dt>
      <dd className="readout mt-1 text-[1.25rem] tabular-nums">{value.toLocaleString()}</dd>
    </div>
  );
}

/** A single count-of-total stat with a tooltip, used by the corpus health
    grid - deliberately just a number, not a bar, since these are flags to
    notice rather than proportions to compare at a glance. */
function HealthStat({
  label,
  value,
  total,
  info,
}: {
  label: string;
  value: number;
  total: number;
  info: string;
}) {
  return (
    <div>
      <dt className="eyebrow inline-flex items-center gap-1.5 text-[0.625rem] text-[var(--ink-faint)]">
        {label}
        <InfoTooltip text={info} />
      </dt>
      <dd className="readout mt-1 text-[1.125rem] tabular-nums">
        {value.toLocaleString()}
        <span className="ml-1 text-[0.75rem] text-[var(--ink-faint)]">/ {total.toLocaleString()}</span>
      </dd>
    </div>
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

/** "Did the last gap detection batch run, when, and what did it find" -
    a summary line, not the full history (see the gap detection tab on the
    admin page for that). Distinct from the pending/approved/rejected
    counts right below it: those describe review state, this describes
    whether the detection job itself ran and succeeded. */
function GapDetectionLastRun({ runs }: { runs: PipelineRun[] }) {
  const last = runs[0];
  if (!last) {
    return <p className="text-[0.8125rem] text-[var(--ink-faint)]">Gap detection has never been run.</p>;
  }

  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[0.8125rem]">
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">last detection run</span>
      <span className="readout">{last.status}</span>
      <span className="text-[var(--ink-faint)]">{new Date(last.started_at).toLocaleString()}</span>
      <span className="text-[var(--ink-soft)]">
        {Object.entries(last.counts)
          .map(([key, value]) => `${key.replace(/_/g, " ")}: ${value.toLocaleString()}`)
          .join(" · ")}
      </span>
      {last.error_summary && <span className="text-[var(--live)]">{last.error_summary}</span>}
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
