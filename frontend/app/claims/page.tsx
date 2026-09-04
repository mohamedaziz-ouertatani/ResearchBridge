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

      <p className="mt-2 max-w-[68ch] whitespace-pre-line font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
        {claim.claim_text}
      </p>

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
