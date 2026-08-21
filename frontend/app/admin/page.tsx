"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { adminApi, type PipelineRun, type PipelineStatus } from "@/lib/adminApi";

/*
  Pipeline status: read-only visibility into IngestionRun/ExtractionRun/
  EmbeddingRun, which existed since the earliest migrations but were never
  exposed anywhere outside direct SQL. Deliberately lightweight - run-level
  summary counts only, no per-record error drill-down, no charts, no
  auto-refresh. Detection/extraction/embedding stay CLI-triggered; this page
  only reads what already ran.
*/

export default function AdminPipeline() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    adminApi
      .pipelineStatus()
      .then(setStatus)
      .catch(() => setError("Can't reach the API. Is it running on port 8000?"));
  }, []);

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
      </header>

      <div className="pt-12">
        <span className="eyebrow">pipeline status</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Read-only visibility into ingestion, extraction, and embedding runs. Detection itself is
          still triggered from the CLI - this only shows what already ran.
        </p>

        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}
        {!status && !error && <p className="eyebrow py-16">loading…</p>}

        {status && (
          <>
            <dl className="mt-8 flex flex-wrap gap-x-10 gap-y-3 border-b border-[var(--rule)] pb-8">
              <Stat label="total papers" value={status.total_papers} />
              <Stat label="with extracted claims" value={status.papers_with_claims} />
              <Stat label="with embeddings" value={status.papers_with_embeddings} />
            </dl>

            <RunSection title="ingestion runs" runs={status.ingestion_runs} />
            <RunSection title="extraction runs" runs={status.extraction_runs} />
            <RunSection title="embedding runs" runs={status.embedding_runs} />
          </>
        )}
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="readout mt-1 text-[1.25rem] tabular-nums">{value.toLocaleString()}</dd>
    </div>
  );
}

function RunSection({ title, runs }: { title: string; runs: PipelineRun[] }) {
  return (
    <section className="border-b border-[var(--rule-soft)] py-8">
      <span className="eyebrow">{title}</span>

      {runs.length === 0 ? (
        <p className="mt-3 text-[0.875rem] text-[var(--ink-faint)]">No runs recorded yet.</p>
      ) : (
        <ul className="mt-4 space-y-4">
          {runs.map((run) => (
            <li key={run.id} className="border-t border-[var(--rule-soft)] pt-4 first:border-t-0 first:pt-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <span className="readout text-[0.875rem]">{run.status}</span>
                <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                  {new Date(run.started_at).toLocaleString()}
                  {run.finished_at
                    ? ` – ${new Date(run.finished_at).toLocaleTimeString()}`
                    : run.status === "running"
                      ? " (in progress)"
                      : ""}
                </span>
              </div>

              <p className="mt-2 text-[0.8125rem] text-[var(--ink-soft)]">
                {Object.entries(run.counts)
                  .map(([key, value]) => `${key.replace(/_/g, " ")}: ${value.toLocaleString()}`)
                  .join(" · ")}
              </p>

              {run.error_summary && (
                <p className="mt-2 text-[0.8125rem] text-[var(--live)]">{run.error_summary}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
