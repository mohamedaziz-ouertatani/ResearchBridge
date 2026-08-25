"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { adminApi, type PipelineKey, type PipelineRun, type PipelineStatus } from "@/lib/adminApi";
import { AdminStats } from "@/components/AdminStats";

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

  One pipeline is shown at a time via tabs rather than all five stacked or
  gridded at once - each tab still carries a "running" dot even when not
  selected, so switching away from a running pipeline doesn't hide that
  it's still going.
*/

type PipelineTab = "arxiv" | "springer" | "semantic_scholar" | "extraction" | "embedding";
type Tab = PipelineTab | "stats";

const PIPELINE_TAB_LABELS: Record<PipelineTab, string> = {
  arxiv: "arXiv",
  springer: "Springer Nature",
  semantic_scholar: "Semantic Scholar",
  extraction: "extraction",
  embedding: "embedding",
};

function runningKeyForTab(tab: PipelineTab): PipelineKey {
  return tab === "extraction" || tab === "embedding" ? tab : (`ingestion_${tab}` as PipelineKey);
}

// Preset topics per source, matching the stratification buckets used
// elsewhere in the corpus (benchmark/domains.py) - each source has its own
// query syntax (arXiv's cat: codes vs free-text OR/|), so the values differ
// even though the topics line up. "Other…" always reveals a free-text field
// for anything not covered here.
const ARXIV_QUERY_OPTIONS = [
  { label: "Machine Learning", value: "cat:cs.LG OR cat:stat.ML" },
  { label: "NLP", value: "cat:cs.CL" },
  { label: "Computer Vision", value: "cat:cs.CV OR cat:eess.IV" },
  { label: "Systems", value: "cat:cs.DC OR cat:cs.OS OR cat:cs.NI OR cat:cs.DB OR cat:cs.SE OR cat:cs.AR OR cat:cs.PF OR cat:cs.DS" },
  { label: "General AI", value: "cat:cs.AI" },
];

const SPRINGER_QUERY_OPTIONS = [
  { label: "Machine Learning", value: '"machine learning"' },
  { label: "NLP", value: '"natural language processing"' },
  { label: "Computer Vision", value: '"computer vision"' },
  { label: "Systems", value: '"distributed systems" OR "computer systems"' },
  { label: "Artificial Intelligence", value: '"artificial intelligence"' },
];

const SEMANTIC_SCHOLAR_QUERY_OPTIONS = [
  { label: "Machine Learning", value: '"machine learning"' },
  { label: "NLP", value: '"natural language processing"' },
  { label: "Computer Vision", value: '"computer vision"' },
  { label: "Systems", value: '"distributed systems" | "computer systems"' },
  { label: "Artificial Intelligence", value: '"artificial intelligence"' },
];

// Mirrors extraction/cli.py's EXTRACTOR_NAMES - kept as a literal list here
// since the frontend can't import the Python tuple, so this needs updating
// by hand if a new Extractor is ever added there.
const EXTRACTOR_OPTIONS = [
  { label: "hybrid (default)", value: "hybrid" },
  { label: "semantic", value: "semantic" },
  { label: "heuristic", value: "heuristic" },
  { label: "stub (synthetic test data only)", value: "stub" },
];

export default function AdminPipeline() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("stats");

  function reload() {
    adminApi
      .pipelineStatus()
      .then((next) => {
        setStatus(next);
        setError(null);
      })
      .catch(() => setError("Can't reach the API. Is it running on port 8000?"));
  }

  // Auto-refresh so run history, "running now" dots, and the stats tab stay
  // current without a manual page reload. 4s while something's running (the
  // same cadence as the per-tab log poll below) so progress reads as live;
  // 15s at idle so a run started from elsewhere (another tab, the CLI) still
  // shows up promptly without polling harder than an idle page needs.
  useEffect(() => {
    reload();
    const isRunning = status ? Object.values(status.running).some(Boolean) : false;
    const interval = setInterval(reload, isRunning ? 4000 : 15000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status ? Object.values(status.running).some(Boolean) : false]);

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
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
            <div className="mt-8 flex flex-wrap gap-x-12 gap-y-6 border-b border-[var(--rule)] pb-8">
              <dl className="flex flex-wrap gap-x-10 gap-y-3">
                <Stat label="total papers" value={status.total_papers} />
                <Stat label="with extracted claims" value={status.papers_with_claims} />
                <Stat label="with embeddings" value={status.papers_with_embeddings} />
              </dl>

              <dl className="flex flex-wrap gap-x-6 gap-y-3">
                {Object.entries(status.papers_by_source).map(([source, count]) => (
                  <Stat key={source} label={source.replace(/_/g, " ")} value={count} small />
                ))}
              </dl>

              <dl className="flex flex-wrap gap-x-10 gap-y-3">
                <Stat label="assessments" value={status.assessment_stats.total} />
                <Stat label="need review" value={status.assessment_stats.needs_review} />
              </dl>
            </div>

            <div className="mt-8 flex flex-wrap gap-1 border-b border-[var(--rule)]">
              <button
                onClick={() => setTab("stats")}
                className={`eyebrow -mb-px px-3 py-2 ${
                  tab === "stats"
                    ? "border-b-2 border-[var(--ink)] text-[var(--ink)]"
                    : "border-b-2 border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
                }`}
              >
                stats
              </button>
              {(Object.keys(PIPELINE_TAB_LABELS) as PipelineTab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={`eyebrow -mb-px flex items-center gap-2 border-b-2 px-3 py-2 ${
                    tab === t
                      ? "border-[var(--ink)] text-[var(--ink)]"
                      : "border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
                  }`}
                >
                  {status.running[runningKeyForTab(t)] && (
                    <span className="h-1.5 w-1.5 rounded-full bg-[var(--live)]" title="running now" />
                  )}
                  {PIPELINE_TAB_LABELS[t]}
                </button>
              ))}
            </div>

            <div className="mt-6">
              {tab === "arxiv" && (
                <RunSection
                  title="arXiv ingestion"
                  pipelineKey="ingestion_arxiv"
                  runs={status.ingestion_runs.filter((run) => run.source === "arxiv")}
                  running={status.running.ingestion_arxiv}
                  fields={[
                    {
                      name: "search_query",
                      label: "search query",
                      type: "select",
                      options: ARXIV_QUERY_OPTIONS,
                      placeholder: "cat:cs.LG OR cat:cs.AI",
                    },
                    { name: "page_size", label: "page size", type: "number" },
                    { name: "max_pages", label: "max pages", type: "number" },
                  ]}
                  onRun={(values) => adminApi.triggerArxivIngestion(values)}
                  onStarted={reload}
                />
              )}

              {tab === "springer" && (
                <RunSection
                  title="Springer Nature ingestion"
                  pipelineKey="ingestion_springer"
                  runs={status.ingestion_runs.filter((run) => run.source === "springer")}
                  running={status.running.ingestion_springer}
                  fields={[
                    {
                      name: "query",
                      label: "query",
                      type: "select",
                      options: SPRINGER_QUERY_OPTIONS,
                      placeholder: '"machine learning" OR "artificial intelligence"',
                    },
                    { name: "page_size", label: "page size", type: "number" },
                    { name: "max_pages", label: "max pages", type: "number" },
                  ]}
                  onRun={(values) => adminApi.triggerSpringerIngestion(values)}
                  onStarted={reload}
                />
              )}

              {tab === "semantic_scholar" && (
                <RunSection
                  title="Semantic Scholar ingestion"
                  pipelineKey="ingestion_semantic_scholar"
                  runs={status.ingestion_runs.filter((run) => run.source === "semantic_scholar")}
                  running={status.running.ingestion_semantic_scholar}
                  fields={[
                    {
                      name: "query",
                      label: "query",
                      type: "select",
                      options: SEMANTIC_SCHOLAR_QUERY_OPTIONS,
                      placeholder: '"machine learning" | "artificial intelligence"',
                    },
                    { name: "max_pages", label: "max pages", type: "number" },
                  ]}
                  onRun={(values) => adminApi.triggerSemanticScholarIngestion(values)}
                  onStarted={reload}
                />
              )}

              {tab === "extraction" && (
                <RunSection
                  title="extraction runs"
                  pipelineKey="extraction"
                  runs={status.extraction_runs}
                  running={status.running.extraction}
                  fields={[
                    { name: "limit", label: "limit", type: "number" },
                    {
                      name: "extractor",
                      label: "extractor",
                      type: "select",
                      options: EXTRACTOR_OPTIONS,
                      placeholder: "hybrid",
                    },
                  ]}
                  onRun={(values) => adminApi.triggerExtraction(values)}
                  onStarted={reload}
                  forceLabel="force re-extract"
                  forceWarning="This deletes every extracted claim and evidence quote in the corpus (and any candidate gap or assessment citing that evidence), then re-runs extraction from scratch. This cannot be undone."
                />
              )}

              {tab === "embedding" && (
                <RunSection
                  title="embedding runs"
                  pipelineKey="embedding"
                  runs={status.embedding_runs}
                  running={status.running.embedding}
                  fields={[{ name: "limit", label: "limit", type: "number" }]}
                  onRun={(values) => adminApi.triggerEmbedding(values)}
                  onStarted={reload}
                  forceLabel="force re-embed"
                  forceWarning="This deletes every existing embedding for the current model, then re-embeds the whole corpus from scratch. This cannot be undone."
                />
              )}

              {tab === "stats" && <AdminStats status={status} />}
            </div>
          </>
        )}
      </div>
    </main>
  );
}

function Stat({ label, value, small = false }: { label: string; value: number; small?: boolean }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className={`readout mt-1 tabular-nums ${small ? "text-[1rem]" : "text-[1.25rem]"}`}>
        {value.toLocaleString()}
      </dd>
    </div>
  );
}

const OTHER_VALUE = "__other__";

const FIELD_INPUT_CLASS =
  "w-full min-w-[6rem] border-b border-[var(--rule)] bg-transparent py-1 text-[0.8125rem] text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:border-[var(--ink)] focus:outline-none disabled:opacity-50";

type Field = {
  name: string;
  label: string;
  type?: "text" | "number" | "select";
  placeholder?: string;
  /** Only for type: "select" - rendered as a dropdown with these choices
   * plus an always-present "Other…" entry that reveals a free-text input
   * bound to the same field, so picking a preset or typing a custom value
   * both just set values[field.name]. */
  options?: { label: string; value: string }[];
};

function RunSection({
  title,
  pipelineKey,
  runs,
  running,
  fields,
  onRun,
  onStarted,
  forceLabel,
  forceWarning,
}: {
  title: string;
  pipelineKey: PipelineKey;
  runs: PipelineRun[];
  running: boolean;
  fields: Field[];
  onRun: (values: Record<string, string | number | boolean>) => Promise<unknown>;
  onStarted: () => void;
  /** When set, renders a second "force re-run" button that wipes prior
   * results before re-running - gated behind an inline confirmation step
   * rather than firing on click, since it's destructive and irreversible. */
  forceLabel?: string;
  forceWarning?: string;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [otherMode, setOtherMode] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [confirmingForce, setConfirmingForce] = useState(false);

  // Poll the running subprocess's log while it's going, so the operator can
  // see progress instead of just "running now" with no detail. One extra
  // fetch right after `running` flips false catches the last lines before
  // polling stops.
  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const fetchLog = () => adminApi.log(pipelineKey).then((text) => !cancelled && setLog(text));
    fetchLog();
    const interval = setInterval(fetchLog, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [running, pipelineKey]);

  function collectPayload(): Record<string, string | number | boolean> {
    const payload: Record<string, string | number | boolean> = {};
    for (const field of fields) {
      const raw = values[field.name]?.trim();
      if (!raw) continue;
      payload[field.name] = field.type === "number" ? Number(raw) : raw;
    }
    return payload;
  }

  async function runNow(payload: Record<string, string | number | boolean>) {
    setBusy(true);
    setError(null);
    try {
      await onRun(payload);
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't start the run.");
    } finally {
      setBusy(false);
      setConfirmingForce(false);
    }
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    void runNow(collectPayload());
  }

  function confirmForceRun() {
    void runNow({ ...collectPayload(), force: true });
  }

  async function stopNow() {
    setStopping(true);
    setError(null);
    try {
      await adminApi.stopPipeline(pipelineKey);
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't stop the run.");
    } finally {
      setStopping(false);
    }
  }

  return (
    <section className="flex flex-col">
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
          <label key={field.name} className="flex flex-1 flex-col gap-1">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{field.label}</span>
            {field.type === "select" ? (
              <>
                <select
                  value={otherMode[field.name] ? OTHER_VALUE : (values[field.name] ?? "")}
                  onChange={(e) => {
                    const picked = e.target.value === OTHER_VALUE;
                    setOtherMode((m) => ({ ...m, [field.name]: picked }));
                    setValues((v) => ({ ...v, [field.name]: picked ? "" : e.target.value }));
                  }}
                  disabled={busy || running}
                  className={FIELD_INPUT_CLASS}
                >
                  <option value="" disabled>
                    choose…
                  </option>
                  {field.options?.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                  <option value={OTHER_VALUE}>Other…</option>
                </select>
                {otherMode[field.name] && (
                  <input
                    type="text"
                    placeholder={field.placeholder}
                    value={values[field.name] ?? ""}
                    onChange={(e) => setValues((v) => ({ ...v, [field.name]: e.target.value }))}
                    disabled={busy || running}
                    className={`mt-1 ${FIELD_INPUT_CLASS}`}
                  />
                )}
              </>
            ) : (
              <input
                type={field.type ?? "text"}
                placeholder={field.placeholder}
                value={values[field.name] ?? ""}
                onChange={(e) => setValues((v) => ({ ...v, [field.name]: e.target.value }))}
                disabled={busy || running}
                className={FIELD_INPUT_CLASS}
              />
            )}
          </label>
        ))}
        <button
          type="submit"
          disabled={busy || running}
          className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
        >
          {busy ? "starting…" : running ? "running…" : "run"}
        </button>

        {running && (
          <button
            type="button"
            onClick={() => void stopNow()}
            disabled={stopping}
            className="eyebrow rounded-[2px] border border-[var(--live)] px-3 py-1.5 text-[var(--live)] hover:bg-[var(--live)] hover:text-white disabled:opacity-50"
          >
            {stopping ? "stopping…" : "stop"}
          </button>
        )}

        {forceLabel && !confirmingForce && (
          <button
            type="button"
            onClick={() => setConfirmingForce(true)}
            disabled={busy || running}
            className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 text-[var(--live)] hover:border-[var(--live)] disabled:opacity-50"
          >
            {forceLabel}
          </button>
        )}
      </form>
      {error && <p className="mt-2 text-[0.75rem] text-[var(--live)]">{error}</p>}

      {forceLabel && confirmingForce && (
        <div className="mt-3 border border-[var(--live)] bg-[var(--panel)] px-4 py-3">
          <p className="text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">{forceWarning}</p>
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={confirmForceRun}
              disabled={busy || running}
              className="eyebrow rounded-[2px] border border-[var(--live)] bg-[var(--live)] px-3 py-1.5 text-white disabled:opacity-50"
            >
              {busy ? "starting…" : `yes, ${forceLabel}`}
            </button>
            <button
              type="button"
              onClick={() => setConfirmingForce(false)}
              disabled={busy}
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
            >
              cancel
            </button>
          </div>
        </div>
      )}

      {running && (
        <div className="mt-4">
          <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">live log</span>
          <pre className="mt-1 max-h-[16rem] overflow-y-auto whitespace-pre-wrap rounded-[2px] border border-[var(--rule-soft)] bg-[var(--panel)] p-3 font-[family-name:var(--type-mono)] text-[0.75rem] leading-relaxed text-[var(--ink-soft)]">
            {log ?? "waiting for output…"}
          </pre>
        </div>
      )}

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
                <p
                  className="mt-2 line-clamp-3 [overflow-wrap:anywhere] text-[0.8125rem] text-[var(--live)]"
                  title={run.error_summary}
                >
                  {run.error_summary}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
