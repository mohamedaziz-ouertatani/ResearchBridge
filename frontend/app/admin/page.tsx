"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  adminApi,
  type ExtractionEvalResult,
  type PipelineKey,
  type PipelineRun,
  type PipelineStatus,
  type RetrievalEvalResult,
} from "@/lib/adminApi";
import { gapsApi } from "@/lib/gapsApi";
import { AdminStats } from "@/components/AdminStats";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { SkeletonStats } from "@/components/Skeleton";

/*
  Pipeline status + triggers. IngestionRun/ExtractionRun/EmbeddingRun existed
  since the earliest migrations but were never exposed anywhere outside
  direct SQL; this page shows run history AND lets an operator start a new
  run of most pipelines (four ingestion sources, extraction, embedding,
  retrieval evaluation) as a background subprocess of the same CLI commands
  (rb-ingest/rb-ingest-springer/rb-ingest-semantic-scholar/rb-ingest-core/
  rb-extract/rb-embed/rb-retrieval-evaluate) - a button, not a new execution
  engine. Gap detection stays a deliberate CLI-only step (see
  gaps_routes.py) - not triggerable here, by design.

  Retrieval evaluation and extraction evaluation have no *_runs history
  table (both are one-off diagnostics, not repeating pipeline stages -
  see RetrievalEvalOut's/ExtractionEvalOut's docstrings), so their tabs
  show only the live run controls plus the last persisted result, not a
  run history list. Citation management used to work the same way but
  now has a real citation_fetch_runs history table, same as ingestion/
  extraction/embedding.

  One pipeline is shown at a time via tabs rather than all of them stacked
  or gridded at once - each tab still carries a "running" dot even when not
  selected, so switching away from a running pipeline doesn't hide that
  it's still going.
*/

type PipelineTab =
  | "arxiv"
  | "springer"
  | "semantic_scholar"
  | "core"
  | "extraction"
  | "embedding"
  | "retrieval_eval"
  | "extraction_eval"
  | "citations_fetch"
  | "gap_detection"
  | "fulltext";
type Tab = PipelineTab | "stats";

const PIPELINE_TAB_LABELS: Record<PipelineTab, string> = {
  arxiv: "arXiv",
  springer: "Springer Nature",
  semantic_scholar: "Semantic Scholar",
  core: "CORE",
  extraction: "extraction",
  embedding: "embedding",
  retrieval_eval: "retrieval eval",
  extraction_eval: "extraction eval",
  citations_fetch: "citation management",
  gap_detection: "gap detection",
  fulltext: "full text",
};

const INGESTION_TABS: PipelineTab[] = ["arxiv", "springer", "semantic_scholar", "core"];

// The four ingestion sources and the two one-off evaluation diagnostics
// each fold behind their own disclosure - same pattern as Nav's
// "infrastructure ▾" menu - so the bar's direct tabs are just the corpus-
// building chain an operator actually watches run-to-run (stats,
// extraction, embedding, full text, citation management, gap detection).
// Twelve flat tabs previously wrapped into an undifferentiated pile; six
// direct tabs plus two labeled dropdowns fit in one row and read as what
// they are: one pipeline, with sources feeding in and diagnostics off to
// the side.
const DIRECT_TABS: PipelineTab[] = ["extraction", "embedding", "fulltext", "citations_fetch", "gap_detection"];
const TAB_DROPDOWNS: { label: string; tabs: PipelineTab[] }[] = [
  { label: "ingestion", tabs: ["arxiv", "springer", "semantic_scholar", "core"] },
  { label: "evaluation", tabs: ["retrieval_eval", "extraction_eval"] },
];

// Every tab's pipeline key matches its tab name 1:1 except the four
// ingestion sources (prefixed "ingestion_") and gap detection, whose key
// is "gaps" - gaps_routes.py's PIPELINE_KEY, set before this page ever
// monitored it, not renamed to match this tab's name.
function runningKeyForTab(tab: PipelineTab): PipelineKey {
  if (INGESTION_TABS.includes(tab)) return `ingestion_${tab}` as PipelineKey;
  if (tab === "gap_detection") return "gaps";
  return tab as PipelineKey;
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

const CORE_QUERY_OPTIONS = [
  { label: "Machine Learning", value: "machine learning" },
  { label: "NLP", value: "natural language processing" },
  { label: "Computer Vision", value: "computer vision" },
  { label: "Systems", value: "distributed systems OR computer systems" },
  { label: "Artificial Intelligence", value: "artificial intelligence" },
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

// Narrower than EXTRACTOR_OPTIONS above: rb-extract-evaluate's --extractor
// only accepts heuristic/semantic/hybrid/all (no "stub" - evaluating the
// synthetic-data stub extractor against real ground truth isn't meaningful).
const EXTRACTION_EVAL_EXTRACTOR_OPTIONS = [
  { label: "all (default)", value: "all" },
  { label: "hybrid", value: "hybrid" },
  { label: "semantic", value: "semantic" },
  { label: "heuristic", value: "heuristic" },
];

const CITATION_SOURCE_OPTIONS = [
  { label: "Semantic Scholar (default)", value: "semantic_scholar" },
  { label: "CrossRef", value: "crossref" },
];

export default function AdminPipeline() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [retrievalEval, setRetrievalEval] = useState<RetrievalEvalResult | null>(null);
  const [extractionEval, setExtractionEval] = useState<ExtractionEvalResult | null>(null);
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
    adminApi.retrievalEval().then(setRetrievalEval).catch(() => {});
    adminApi.extractionEval().then(setExtractionEval).catch(() => {});
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
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">pipeline status</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Run history for ingestion, extraction, embedding, citation fetching, and gap detection,
          plus a way to start a new run of each. Gap detection can also be triggered from the{" "}
          <Link href="/gaps" className="underline hover:text-[var(--ink)]">
            candidate gaps
          </Link>{" "}
          page — both trigger the same background job, so either one shows the other&apos;s progress.
        </p>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          The corpus is built by a chain of stages: ingestion pulls raw papers in from an external
          source (arXiv, Springer Nature, Semantic Scholar, or CORE), extraction pulls structured claims
          and evidence out of each paper&apos;s text, embedding turns that text into the vectors search
          relies on, and gap detection compares each paper&apos;s neighborhood for recurring
          limitation patterns. Each tab below runs and monitors one stage; a paper generally needs
          to pass through extraction and embedding before it&apos;s eligible for gap detection.
        </p>

        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}
        {!status && !error && <SkeletonStats count={5} />}

        {status && (
          <>
            <div className="mt-8 flex flex-wrap gap-x-12 gap-y-6 border-b border-[var(--rule)] pb-8">
              <dl className="flex flex-wrap gap-x-10 gap-y-3">
                <Stat
                  label="total papers"
                  value={status.total_papers}
                  info="Every paper currently in the corpus, across all three sources, regardless of what stage of processing it's reached."
                />
                <Stat
                  label="with extracted claims"
                  value={status.papers_with_claims}
                  info="Papers that have been through the extraction stage, so their claims and evidence are available for assessments and gap detection."
                />
                <Stat
                  label="with embeddings"
                  value={status.papers_with_embeddings}
                  info="Papers that have been through the embedding stage, so they're searchable by meaning and eligible for gap detection's neighborhood comparisons."
                />
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

            <div className="mt-8 flex flex-wrap items-center gap-x-1 gap-y-2 border-b border-[var(--rule)]">
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

              <TabDropdown
                {...TAB_DROPDOWNS[0]}
                tab={tab}
                runningByTab={status.running}
                runningKeyForTab={runningKeyForTab}
                onSelect={setTab}
              />

              {DIRECT_TABS.map((t) => (
                <TabButton key={t} tab={t} active={tab === t} running={status.running[runningKeyForTab(t)]} onSelect={setTab} />
              ))}

              <TabDropdown
                {...TAB_DROPDOWNS[1]}
                tab={tab}
                runningByTab={status.running}
                runningKeyForTab={runningKeyForTab}
                onSelect={setTab}
              />
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

              {tab === "core" && (
                <RunSection
                  title="CORE ingestion"
                  pipelineKey="ingestion_core"
                  runs={status.ingestion_runs.filter((run) => run.source === "core")}
                  running={status.running.ingestion_core}
                  fields={[
                    {
                      name: "query",
                      label: "query",
                      type: "select",
                      options: CORE_QUERY_OPTIONS,
                      placeholder: "machine learning OR artificial intelligence",
                    },
                    { name: "page_size", label: "page size", type: "number" },
                    { name: "max_pages", label: "max pages", type: "number" },
                  ]}
                  onRun={(values) => adminApi.triggerCoreIngestion(values)}
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

              {tab === "retrieval_eval" && (
                <>
                  <RunSection
                    title="retrieval evaluation"
                    pipelineKey="retrieval_eval"
                    runs={[]}
                    running={status.running.retrieval_eval}
                    fields={[{ name: "k", label: "k", type: "number", placeholder: "10" }]}
                    onRun={(values) => adminApi.triggerRetrievalEval(values)}
                    onStarted={reload}
                  />
                  <RetrievalEvalResults result={retrievalEval} />
                </>
              )}

              {tab === "extraction_eval" && (
                <>
                  <RunSection
                    title="extraction evaluation"
                    pipelineKey="extraction_eval"
                    runs={[]}
                    running={status.running.extraction_eval}
                    fields={[
                      { name: "threshold", label: "similarity threshold", type: "number", placeholder: "0.5" },
                      {
                        name: "extractor",
                        label: "extractor",
                        type: "select",
                        options: EXTRACTION_EVAL_EXTRACTOR_OPTIONS,
                        placeholder: "all",
                      },
                    ]}
                    onRun={(values) => adminApi.triggerExtractionEval(values)}
                    onStarted={reload}
                  />
                  <ExtractionEvalResults result={extractionEval} />
                </>
              )}

              {tab === "citations_fetch" && (
                <>
                  <RunSection
                    title="citation management"
                    pipelineKey="citations_fetch"
                    runs={status.citation_fetch_runs}
                    running={status.running.citations_fetch}
                    fields={[
                      {
                        name: "source",
                        label: "source",
                        type: "select",
                        options: CITATION_SOURCE_OPTIONS,
                        placeholder: "semantic_scholar",
                      },
                    ]}
                    onRun={(values) => adminApi.triggerCitationsFetch(values)}
                    onStarted={reload}
                    forceLabel="force refetch"
                    forceWarning="This reprocesses every paper eligible for the selected source, including ones that already have citation edges from it, instead of only papers without any yet. It won't delete existing edges, but re-fetches for the whole corpus again, which takes a long time and uses many more API calls."
                  />
                </>
              )}

              {tab === "gap_detection" && (
                <RunSection
                  title="gap detection runs"
                  pipelineKey="gaps"
                  runs={status.gap_detection_runs}
                  running={status.running.gaps}
                  fields={[]}
                  onRun={() => gapsApi.detect()}
                  onStarted={reload}
                />
              )}

              {tab === "fulltext" && (
                <>
                  <p className="mb-4 max-w-[60ch] text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
                    Fetches and parses each open-access paper&apos;s full text (arXiv, Semantic Scholar,
                    Springer via PDF; CORE via its own API, since its PDF downloads are Cloudflare-blocked).
                    Extraction (on its next run for a given paper) prefers full-text sections here over
                    the abstract when they&apos;re available. arXiv fetches are throttled to one every 3
                    seconds, so a full run against the whole corpus can take hours; use limit to try a
                    small batch first.
                  </p>
                  <RunSection
                    title="full-text fetch runs"
                    pipelineKey="fulltext"
                    runs={status.fulltext_fetch_runs}
                    running={status.running.fulltext}
                    fields={[{ name: "limit", label: "limit", type: "number" }]}
                    onRun={(values) => adminApi.triggerFulltext(values)}
                    onStarted={reload}
                    forceLabel="force re-fetch"
                    forceWarning="Re-fetches papers that already have full text, overwriting it with a fresh fetch, instead of only papers that don't have any yet. Not destructive - each paper's row is updated in place - but re-processes the whole eligible set again, which takes a long time (arXiv alone is throttled to one paper every 3 seconds)."
                  />
                </>
              )}

              {tab === "stats" && <AdminStats status={status} />}
            </div>
          </>
        )}
      </div>
    </main>
  );
}

/** A labeled disclosure holding a few related tabs (ingestion sources,
    evaluation diagnostics) - same open/close-on-outside-click behavior as
    Nav's InfrastructureMenu, adapted from links to in-page tab selection.
    Shows "running now" on the trigger itself when a folded tab is
    running, so progress on a collapsed pipeline is never hidden. */
function TabDropdown({
  label,
  tabs,
  tab,
  runningByTab,
  runningKeyForTab,
  onSelect,
}: {
  label: string;
  tabs: PipelineTab[];
  tab: Tab;
  runningByTab: Record<PipelineKey, boolean>;
  runningKeyForTab: (tab: PipelineTab) => PipelineKey;
  onSelect: (tab: PipelineTab) => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const active = tabs.includes(tab as PipelineTab);
  const anyRunning = tabs.some((t) => runningByTab[runningKeyForTab(t)]);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={`eyebrow -mb-px flex items-center gap-1.5 border-b-2 px-3 py-2 ${
          active
            ? "border-[var(--ink)] text-[var(--ink)]"
            : "border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
        }`}
      >
        {anyRunning && <span className="h-1.5 w-1.5 rounded-full bg-[var(--live)]" title="running now" />}
        {label}
        <span aria-hidden className={`text-[0.625rem] transition-transform ${open ? "rotate-180" : ""}`}>
          ▾
        </span>
      </button>

      {open && (
        <div className="absolute top-[calc(100%+0.4rem)] left-0 z-10 w-[14rem] border border-[var(--rule)] bg-[var(--panel)] py-1 shadow-lg">
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => {
                onSelect(t);
                setOpen(false);
              }}
              className={`flex w-full items-center gap-2 px-3 py-2 text-left text-[0.8125rem] ${
                tab === t ? "bg-[var(--field)] text-[var(--ink)]" : "text-[var(--ink-soft)] hover:bg-[var(--field)]"
              }`}
            >
              {runningByTab[runningKeyForTab(t)] && (
                <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--live)]" title="running now" />
              )}
              {PIPELINE_TAB_LABELS[t]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function TabButton({
  tab,
  active,
  running,
  onSelect,
}: {
  tab: PipelineTab;
  active: boolean;
  running: boolean;
  onSelect: (tab: PipelineTab) => void;
}) {
  return (
    <button
      onClick={() => onSelect(tab)}
      className={`eyebrow -mb-px flex items-center gap-2 border-b-2 px-3 py-2 ${
        active
          ? "border-[var(--ink)] text-[var(--ink)]"
          : "border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
      }`}
    >
      {running && <span className="h-1.5 w-1.5 rounded-full bg-[var(--live)]" title="running now" />}
      {PIPELINE_TAB_LABELS[tab]}
    </button>
  );
}

function Stat({
  label,
  value,
  small = false,
  info,
}: {
  label: string;
  value: number;
  small?: boolean;
  info?: string;
}) {
  return (
    <div>
      <dt className="eyebrow inline-flex items-center gap-1.5">
        {label}
        {info && <InfoTooltip text={info} />}
      </dt>
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
        <span className="eyebrow inline-flex items-center gap-1.5">
          {title}
          <InfoTooltip text="Fill in the fields and click run to start this pipeline as a background job. It keeps running even if you navigate away; come back to this tab to check progress or stop it. “Running now” and the live log also pick up a run started directly from a terminal (not just this button), as long as it's still alive." />
        </span>
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
          <>
            <button
              type="button"
              onClick={() => setConfirmingForce(true)}
              disabled={busy || running}
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 text-[var(--live)] hover:border-[var(--live)] disabled:opacity-50"
            >
              {forceLabel}
            </button>
            <InfoTooltip
              label="What does force re-run do?"
              text="Unlike the regular run button, this deletes existing results for this stage first and rebuilds them from scratch instead of only processing what's new. It's destructive and cannot be undone, so it asks for confirmation before starting."
            />
          </>
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
                <span className="readout inline-flex items-center gap-2 text-[0.875rem]">
                  {run.status}
                  {run.source && (
                    <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                      {run.source.replace(/_/g, " ")}
                    </span>
                  )}
                  {run.pid != null && (
                    <span
                      className="eyebrow text-[0.625rem] text-[var(--ink-faint)]"
                      title="OS process id - correlate with `ps` or Task Manager, e.g. to stop a run started outside this panel"
                    >
                      pid {run.pid}
                    </span>
                  )}
                </span>
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

/** The last rb-retrieval-evaluate run's persisted metrics - never computed
    live here, just read (see adminApi.retrievalEval). One table per query
    set (self/topical), one row per baseline (tfidf/bm25/embedding/hybrid). */
function RetrievalEvalResults({ result }: { result: RetrievalEvalResult | null }) {
  if (!result) return null;

  if (!result.available) {
    return (
      <p className="mt-8 text-[0.9375rem] text-[var(--ink-soft)]">
        Never run yet - click run to evaluate retrieval quality against the benchmark.
      </p>
    );
  }

  return (
    <div className="mt-8 space-y-8">
      <p className="text-[0.8125rem] text-[var(--ink-faint)]">
        Last run {result.generated_at ? new Date(result.generated_at).toLocaleString() : "—"}, k={result.k}
      </p>
      {Object.entries(result.query_sets ?? {}).map(([name, querySet]) => (
        <div key={name}>
          <span className="eyebrow">
            {name} query set — {querySet.queries} quer{querySet.queries === 1 ? "y" : "ies"}
            {querySet.skipped > 0 ? `, ${querySet.skipped} skipped` : ""}
          </span>
          {querySet.results.length === 0 ? (
            <p className="mt-2 text-[0.8125rem] text-[var(--ink-faint)]">No usable queries for this set.</p>
          ) : (
            <table className="mt-3 w-full text-[0.8125rem]">
              <thead>
                <tr className="border-b border-[var(--rule)] text-left text-[var(--ink-faint)]">
                  <th className="py-1 pr-4 font-normal">method</th>
                  <th className="py-1 pr-4 font-normal">P@{result.k}</th>
                  <th className="py-1 pr-4 font-normal">R@{result.k}</th>
                  <th className="py-1 pr-4 font-normal">nDCG@{result.k}</th>
                  <th className="py-1 font-normal">MRR</th>
                </tr>
              </thead>
              <tbody>
                {querySet.results.map((row) => (
                  <tr key={row.method} className="border-b border-[var(--rule-soft)]">
                    <td className="py-1.5 pr-4 readout">{row.method}</td>
                    <td className="py-1.5 pr-4 tabular-nums">{row.precision.toFixed(3)}</td>
                    <td className="py-1.5 pr-4 tabular-nums">{row.recall.toFixed(3)}</td>
                    <td className="py-1.5 pr-4 tabular-nums">{row.ndcg.toFixed(3)}</td>
                    <td className="py-1.5 tabular-nums">{row.mrr.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      ))}
    </div>
  );
}

/** The last rb-extract-evaluate run's persisted per-field scores - never
    computed live here, just read (see adminApi.extractionEval). One table
    per extractor (heuristic/semantic/hybrid), one row per field. */
function ExtractionEvalResults({ result }: { result: ExtractionEvalResult | null }) {
  if (!result) return null;

  if (!result.available) {
    return (
      <p className="mt-8 text-[0.9375rem] text-[var(--ink-soft)]">
        Never run yet - click run to evaluate extraction quality against the benchmark.
      </p>
    );
  }

  return (
    <div className="mt-8 space-y-8">
      <p className="text-[0.8125rem] text-[var(--ink-faint)]">
        Last run {result.generated_at ? new Date(result.generated_at).toLocaleString() : "—"}, threshold=
        {result.threshold}, {result.paper_count} paper{result.paper_count === 1 ? "" : "s"}
      </p>
      {Object.entries(result.extractors ?? {}).map(([extractorName, fieldScores]) => (
        <div key={extractorName}>
          <span className="eyebrow">{extractorName}</span>
          <table className="mt-3 w-full text-[0.8125rem]">
            <thead>
              <tr className="border-b border-[var(--rule)] text-left text-[var(--ink-faint)]">
                <th className="py-1 pr-4 font-normal">field</th>
                <th className="py-1 pr-4 font-normal">precision</th>
                <th className="py-1 pr-4 font-normal">recall</th>
                <th className="py-1 font-normal">f1</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(fieldScores).map(([field, score]) => (
                <tr key={field} className="border-b border-[var(--rule-soft)]">
                  <td className="py-1.5 pr-4 readout">{field.replace(/_/g, " ")}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{score.precision.toFixed(3)}</td>
                  <td className="py-1.5 pr-4 tabular-nums">{score.recall.toFixed(3)}</td>
                  <td className="py-1.5 tabular-nums">{score.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

