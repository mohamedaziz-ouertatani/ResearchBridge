"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { adminApi, type PipelineKey, type PipelineRun, type PipelineStatus } from "@/lib/adminApi";

/*
  Pipeline status + triggers. IngestionRun/ExtractionRun/EmbeddingRun existed
  since the earliest migrations but were never exposed anywhere outside
  direct SQL; this page shows run history AND lets an operator start a new
  run of any of the five pipelines (three ingestion sources, extraction,
  embedding) as a background subprocess of the same CLI commands
  (rb-ingest/rb-ingest-springer/rb-ingest-semantic-scholar/rb-extract/
  rb-embed) - a button, not a new execution engine. Gap detection stays a
  deliberate CLI-only step (see gaps_routes.py) - not triggerable here, by
  design.
*/

export default function AdminPipeline() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  function reload() {
    adminApi
      .pipelineStatus()
      .then(setStatus)
      .catch(() => setError("Can't reach the API. Is it running on port 8000?"));
  }

  useEffect(reload, []);

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
          Run history for ingestion, extraction, and embedding, plus a way to start a new run of
          each. Gap detection stays CLI-only (<code className="readout text-[0.8125rem]">rb-gaps-detect</code>)
          - not triggerable here.
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

            <RunSection
              title="arXiv ingestion"
              runs={status.ingestion_runs.filter((run) => run.source === "arxiv")}
              running={status.running.ingestion_arxiv}
              fields={[
                { name: "search_query", label: "search query", placeholder: "cat:cs.LG OR cat:cs.AI" },
                { name: "page_size", label: "page size", type: "number" },
                { name: "max_pages", label: "max pages", type: "number" },
              ]}
              onRun={(values) => adminApi.triggerArxivIngestion(values)}
              onStarted={reload}
            />

            <RunSection
              title="Springer Nature ingestion"
              runs={status.ingestion_runs.filter((run) => run.source === "springer")}
              running={status.running.ingestion_springer}
              fields={[
                { name: "query", label: "query", placeholder: '"machine learning" OR "artificial intelligence"' },
                { name: "page_size", label: "page size", type: "number" },
                { name: "max_pages", label: "max pages", type: "number" },
              ]}
              onRun={(values) => adminApi.triggerSpringerIngestion(values)}
              onStarted={reload}
            />

            <RunSection
              title="Semantic Scholar ingestion"
              runs={status.ingestion_runs.filter((run) => run.source === "semantic_scholar")}
              running={status.running.ingestion_semantic_scholar}
              fields={[
                { name: "query", label: "query", placeholder: '"machine learning" | "artificial intelligence"' },
                { name: "max_pages", label: "max pages", type: "number" },
              ]}
              onRun={(values) => adminApi.triggerSemanticScholarIngestion(values)}
              onStarted={reload}
            />

            <RunSection
              title="extraction runs"
              runs={status.extraction_runs}
              running={status.running.extraction}
              fields={[
                { name: "limit", label: "limit", type: "number" },
                { name: "extractor", label: "extractor", placeholder: "hybrid" },
              ]}
              onRun={(values) => adminApi.triggerExtraction(values)}
              onStarted={reload}
            />

            <RunSection
              title="embedding runs"
              runs={status.embedding_runs}
              running={status.running.embedding}
              fields={[{ name: "limit", label: "limit", type: "number" }]}
              onRun={(values) => adminApi.triggerEmbedding(values)}
              onStarted={reload}
            />
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

type Field = { name: string; label: string; type?: "text" | "number"; placeholder?: string };

function RunSection({
  title,
  runs,
  running,
  fields,
  onRun,
  onStarted,
}: {
  title: string;
  runs: PipelineRun[];
  running: boolean;
  fields: Field[];
  onRun: (values: Record<string, string | number>) => Promise<unknown>;
  onStarted: () => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const payload: Record<string, string | number> = {};
      for (const field of fields) {
        const raw = values[field.name]?.trim();
        if (!raw) continue;
        payload[field.name] = field.type === "number" ? Number(raw) : raw;
      }
      await onRun(payload);
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't start the run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="border-b border-[var(--rule-soft)] py-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <span className="eyebrow">{title}</span>
        <span
          className={`readout text-[0.6875rem] ${running ? "text-[var(--live)]" : "text-[var(--ink-faint)]"}`}
        >
          {running ? "running now" : "idle"}
        </span>
      </div>

      <form onSubmit={submit} className="mt-4 flex flex-wrap items-end gap-3">
        {fields.map((field) => (
          <label key={field.name} className="flex flex-col gap-1">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{field.label}</span>
            <input
              type={field.type ?? "text"}
              placeholder={field.placeholder}
              value={values[field.name] ?? ""}
              onChange={(e) => setValues((v) => ({ ...v, [field.name]: e.target.value }))}
              disabled={busy || running}
              className="w-40 border-b border-[var(--rule)] bg-transparent py-1 text-[0.8125rem] text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:border-[var(--ink)] focus:outline-none disabled:opacity-50"
            />
          </label>
        ))}
        <button
          type="submit"
          disabled={busy || running}
          className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
        >
          {busy ? "starting…" : running ? "running…" : "run"}
        </button>
      </form>
      {error && <p className="mt-2 text-[0.75rem] text-[var(--live)]">{error}</p>}

      {runs.length === 0 ? (
        <p className="mt-6 text-[0.875rem] text-[var(--ink-faint)]">No runs recorded yet.</p>
      ) : (
        <ul className="mt-6 space-y-4">
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
