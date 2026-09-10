"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  claimsApi,
  CLAIM_TYPE_LABELS,
  SOURCE_TABLE_LABELS,
  type AnalysisClaim,
  type ClaimStatusFilter,
  type ClaimTypeFilter,
  type SourceTableFilter,
} from "@/lib/claimsApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { SkeletonRows } from "@/components/Skeleton";

const LIMIT = 20;

const STATUS_OPTIONS: { value: ClaimStatusFilter; label: string }[] = [
  { value: "all", label: "all statuses" },
  { value: "pending", label: "pending" },
  { value: "approved", label: "approved" },
  { value: "rejected", label: "rejected" },
];

const CLAIM_TYPE_OPTIONS: { value: ClaimTypeFilter; label: string }[] = [
  { value: "all", label: "all types" },
  { value: "fact", label: "fact" },
  { value: "inference", label: "inference" },
  { value: "hypothesis", label: "hypothesis" },
  { value: "opportunity", label: "opportunity" },
  { value: "speculation", label: "speculation" },
];

const SOURCE_TABLE_OPTIONS: { value: SourceTableFilter; label: string }[] = [
  { value: "all", label: "gaps + assessments" },
  { value: "candidate_gaps", label: "candidate gaps only" },
  { value: "research_assessments", label: "assessments only" },
];

const SELECT_CLASS =
  "readout border-b border-[var(--rule)] bg-transparent py-1 text-[0.8125rem] text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none";

export default function Claims() {
  const [status, setStatus] = useState<ClaimStatusFilter>("all");
  const [claimType, setClaimType] = useState<ClaimTypeFilter>("all");
  const [sourceTable, setSourceTable] = useState<SourceTableFilter>("all");
  const [offset, setOffset] = useState(0);
  const [claims, setClaims] = useState<AnalysisClaim[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    claimsApi
      .list({ status, claim_type: claimType, source_table: sourceTable }, LIMIT, offset)
      .then((page) => {
        setClaims(page.items);
        setTotal(page.total);
      })
      .catch(() => setError("Couldn't load claims."))
      .finally(() => setLoading(false));
  }, [status, claimType, sourceTable, offset]);

  function resetAndSet<T>(setter: (v: T) => void) {
    return (value: T) => {
      setOffset(0);
      setter(value);
    };
  }

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">claims</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          The Sec 16 structured-reasoning layer: every candidate gap and every gradeable assessment
          field, mirrored as a typed claim (fact/inference/hypothesis/opportunity/speculation) linked
          to the real evidence it was grounded in — the Evidence → Inference chain made inspectable
          on its own, not nested inside a specific gap or assessment page.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">status</span>
            <select
              value={status}
              onChange={(e) => resetAndSet(setStatus)(e.target.value as ClaimStatusFilter)}
              className={SELECT_CLASS}
            >
              {STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">type</span>
            <select
              value={claimType}
              onChange={(e) => resetAndSet(setClaimType)(e.target.value as ClaimTypeFilter)}
              className={SELECT_CLASS}
            >
              {CLAIM_TYPE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">source</span>
            <select
              value={sourceTable}
              onChange={(e) => resetAndSet(setSourceTable)(e.target.value as SourceTableFilter)}
              className={SELECT_CLASS}
            >
              {SOURCE_TABLE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <InfoTooltip text="A claim's status mirrors its source's own review: a candidate gap's approve/reject, or an assessment's human_reviewed flag. It is never reviewed independently here." />
        </div>

        {loading && <SkeletonRows count={4} />}
        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

        {!loading && !error && claims.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">No claims match these filters.</p>
        )}

        {!loading && !error && claims.length > 0 && (
          <>
            <ul>
              {claims.map((claim, i) => (
                <ClaimCard key={claim.id} claim={claim} index={i} />
              ))}
            </ul>

            <div className="mt-6 flex items-center justify-between border-t border-[var(--rule)] pt-4">
              <span className="readout text-[0.75rem] text-[var(--ink-faint)]">
                {offset + 1}–{Math.min(offset + LIMIT, total)} of {total}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setOffset((o) => Math.max(0, o - LIMIT))}
                  disabled={offset === 0}
                  className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-40"
                >
                  previous
                </button>
                <button
                  onClick={() => setOffset((o) => o + LIMIT)}
                  disabled={offset + LIMIT >= total}
                  className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-40"
                >
                  next
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}

// Above this, a claim's text routinely runs several paragraphs (an
// inference claim mirrors a whole novelty_reasoning field verbatim,
// dimension-coverage appendix included - see assessment/claims.py) and
// reads as a wall of text in a list of otherwise short claims. Collapsed
// by default past this length, same "make the reader ask for more" pattern
// as the evidence <details> below it.
const LONG_CLAIM_TEXT_THRESHOLD = 320;

// A "fact" claim mirrors an entire report field verbatim (assessment/
// claims.py - comparison_summary and risks_and_limitations both bundle
// several different papers' own quotes into one claim, deliberately never
// synthesized into a single sentence). Parsed into per-paper bullets here
// so it reads as "several papers say X, Y, Z" instead of one dense
// paragraph of dashes and quotation marks - same content, legible shape.
// Two bullet shapes exist depending on which field produced it:
//   - "Title": unquoted claim text        (comparison_summary)
//   - Title: "quoted claim text"          (risks_and_limitations)
// [\s\S]* (not .*) for the captured claim text so an extracted quote that
// itself wraps mid-sentence - a real, observed shape in extraction output -
// doesn't get cut off at the first embedded newline.
const QUOTED_TITLE_BULLET_RE = /^-\s*"([^"]*)":\s*([\s\S]*)$/;
const QUOTED_TEXT_BULLET_RE = /^-\s*(.+?):\s*"([\s\S]*)"$/;

type FactItem = { title: string; text: string };
type FactBlock = { heading: string | null; items: FactItem[] };

function parseBulletLine(line: string): FactItem | null {
  const quotedTitle = QUOTED_TITLE_BULLET_RE.exec(line);
  if (quotedTitle) return { title: quotedTitle[1], text: quotedTitle[2] };
  const quotedText = QUOTED_TEXT_BULLET_RE.exec(line);
  if (quotedText) return { title: quotedText[1], text: quotedText[2] };
  return null;
}

/** A new bullet starts on a "- " line; any line before the next one is a
 * continuation of it (or of the heading, for the block's first line) - the
 * same wrapped-quote case QUOTED_*_BULLET_RE's [\s\S]* accounts for. */
function groupBulletLines(lines: string[]): string[] {
  const chunks: string[] = [];
  for (const line of lines) {
    if (chunks.length === 0 || /^-\s/.test(line)) {
      chunks.push(line);
    } else {
      chunks[chunks.length - 1] += `\n${line}`;
    }
  }
  return chunks;
}

/** Returns null (render as plain text instead) when claim_text doesn't
 * actually look like the bundled bullet format - never guess. Exported for
 * __tests__/claimsFactParsing.test.ts. */
export function parseFactBlocks(text: string): FactBlock[] | null {
  const blocks = (text.includes("\n\n") ? text.split("\n\n") : [text]).filter(Boolean);
  if (blocks.length === 0) return null;

  const parsed: FactBlock[] = [];
  for (const block of blocks) {
    const chunks = groupBulletLines(block.split("\n").filter(Boolean));
    const heading = parseBulletLine(chunks[0]) ? null : chunks[0];
    const itemChunks = heading ? chunks.slice(1) : chunks;
    if (itemChunks.length === 0) return null;

    const items = itemChunks.map(parseBulletLine);
    if (items.some((item) => item === null)) return null;
    parsed.push({ heading, items: items as FactItem[] });
  }
  return parsed;
}

const FACT_ITEMS_SHOWN_BY_DEFAULT = 3;

/** Truncates a flat item budget across blocks without touching block
 * boundaries or headings - a pure pass, no state, so it's safe to
 * recompute on every render. */
function takeBlocks(blocks: FactBlock[], limit: number | null): FactBlock[] {
  const taken: FactBlock[] = [];
  let budget = limit;
  for (const block of blocks) {
    if (budget !== null && budget <= 0) break;
    const items = budget === null ? block.items : block.items.slice(0, budget);
    if (items.length === 0) continue;
    taken.push({ heading: block.heading, items });
    if (budget !== null) budget -= items.length;
  }
  return taken;
}

function FactClaimText({ text }: { text: string }) {
  const blocks = parseFactBlocks(text);
  const [expanded, setExpanded] = useState(false);

  if (!blocks) return <PlainClaimText text={text} />;

  const totalItems = blocks.reduce((n, b) => n + b.items.length, 0);
  const hasMore = totalItems > FACT_ITEMS_SHOWN_BY_DEFAULT;
  const visibleBlocks = takeBlocks(blocks, expanded ? null : FACT_ITEMS_SHOWN_BY_DEFAULT);

  return (
    <div className="mt-2 max-w-[68ch] space-y-4">
      {visibleBlocks.map((block, blockIndex) => (
        <div key={blockIndex}>
          {block.heading && (
            <p className="font-[family-name:var(--type-text)] text-[0.9375rem] font-semibold leading-snug text-[var(--ink)]">
              {block.heading}
            </p>
          )}
          <ul className={`space-y-2 ${block.heading ? "mt-2" : ""}`}>
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex}>
                <p className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
                  {item.text}
                </p>
                <span className="eyebrow mt-0.5 inline-block text-[var(--ink-faint)]">
                  {item.title}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
      {hasMore && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="eyebrow text-[var(--ink-faint)] hover:text-[var(--ink)]"
        >
          {expanded ? "show fewer" : `show ${totalItems - FACT_ITEMS_SHOWN_BY_DEFAULT} more`}
        </button>
      )}
    </div>
  );
}

function PlainClaimText({ text }: { text: string }) {
  const isLong = text.length > LONG_CLAIM_TEXT_THRESHOLD;
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <p
        className={`mt-2 max-w-[68ch] whitespace-pre-line font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)] ${
          isLong && !expanded ? "line-clamp-4" : ""
        }`}
      >
        {text}
      </p>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="eyebrow mt-1.5 text-[var(--ink-faint)] hover:text-[var(--ink)]"
        >
          {expanded ? "show less" : "show more"}
        </button>
      )}
    </>
  );
}

function ClaimCard({ claim, index }: { claim: AnalysisClaim; index: number }) {
  return (
    <li
      className="resolve border-t border-[var(--rule-soft)] py-6 first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="eyebrow">
          {CLAIM_TYPE_LABELS[claim.claim_type]} · confidence: {claim.confidence}
        </span>
        <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
          {SOURCE_TABLE_LABELS[claim.source_table]} · {claim.status}
        </span>
      </div>

      {claim.claim_type === "fact" ? (
        <FactClaimText text={claim.claim_text} />
      ) : (
        <PlainClaimText text={claim.claim_text} />
      )}

      {claim.evidence.length > 0 && (
        <details className="mt-3">
          <summary className="eyebrow cursor-pointer hover:text-[var(--ink)]">
            supporting evidence ({claim.evidence.length})
          </summary>
          <ul className="mt-2 space-y-3 border-l border-[var(--rule-soft)] pl-4">
            {claim.evidence.map((item, i) => (
              <li key={i}>
                <div className="flex flex-wrap items-center gap-2">
                  <Link href={`/papers/${item.paper_id}`} className="eyebrow hover:text-[var(--ink)]">
                    {item.paper_title}
                  </Link>
                  <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                    {item.relationship}
                    {item.section ? ` · ${item.section}` : ""}
                  </span>
                </div>
                <p className="mt-1 max-w-[62ch] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                  &quot;{item.text}&quot;
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}
