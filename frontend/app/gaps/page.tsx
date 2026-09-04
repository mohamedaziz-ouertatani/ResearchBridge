"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  gapsApi,
  RATING_DIMENSIONS,
  GAP_STATUS_LABELS,
  type CandidateGap,
  type GapRatings,
  type GapStatusFilter,
} from "@/lib/gapsApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { SkeletonRows } from "@/components/Skeleton";

const FILTERS: GapStatusFilter[] = ["pending", "approved", "rejected", "all"];

export default function GapReview() {
  const [status, setStatus] = useState<GapStatusFilter>("pending");
  const [gaps, setGaps] = useState<CandidateGap[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Resetting loading/error before a fetch keyed on `status` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    gapsApi
      .list(status)
      .then((page) => setGaps(page.items))
      .catch(() => setError("Couldn't load candidate gaps."))
      .finally(() => setLoading(false));
  }, [status]);

  const [running, setRunning] = useState(false);
  const [log, setLog] = useState("");
  const [detectError, setDetectError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      gapsApi.detectStatus().then((next) => {
        if (cancelled) return;
        setRunning(next.running);
        setLog(next.log);
      });
    }
    poll();
    const interval = setInterval(poll, running ? 3000 : 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [running]);

  async function handleDetect() {
    setDetectError(null);
    try {
      await gapsApi.detect();
      setRunning(true);
    } catch {
      setDetectError("Couldn't start gap detection — it may already be running.");
    }
  }

  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      gapsApi.list(status).then((page) => setGaps(page.items));
    }
    wasRunning.current = running;
  }, [running, status]);

  function handleReviewed(reviewed: CandidateGap) {
    // reviewing moves a gap out of whatever list it was in (unless viewing "all")
    setGaps((prev) =>
      status === "all" ? prev.map((g) => (g.id === reviewed.id ? reviewed : g)) : prev.filter((g) => g.id !== reviewed.id),
    );
  }

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">candidate gaps</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Recurring limitation/research-gap patterns inferred across a paper&apos;s neighborhood — never a
          single author&apos;s claim, and never presented as validated until reviewed here. Approving or
          rejecting is the only way a candidate leaves &quot;pending&quot;.
        </p>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          A candidate gap is a computer-generated observation: the system noticed that several
          papers near each other in embedding space share the same limitation, and is surfacing
          that pattern for a human to check. This page is that check — nothing here counts as an
          established gap in the literature until someone approves it.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <button
            onClick={handleDetect}
            disabled={running}
            className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
          >
            {running ? "generating…" : "generate gaps"}
          </button>
          <InfoTooltip text="Scans the corpus for new candidate gaps and adds any it finds to the pending queue below. This runs the same detection the CLI uses; it can take a while on a large corpus, and the log below shows its progress." />
          {detectError && <span className="text-[0.75rem] text-[var(--live)]">{detectError}</span>}
        </div>

        {running && log && (
          <pre className="readout mt-3 max-h-40 overflow-auto whitespace-pre-wrap border border-[var(--rule-soft)] p-3 text-[0.75rem] text-[var(--ink-soft)]">
            {log}
          </pre>
        )}

        <div className="mt-6 flex gap-1 border-b border-[var(--rule)]">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setStatus(f)}
              className={`eyebrow -mb-px border-b-2 px-3 py-2 ${
                status === f ? "border-[var(--ink)] text-[var(--ink)]" : "border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        {loading && <SkeletonRows count={3} />}
        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

        {!loading && !error && gaps.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            No {status === "all" ? "" : status} candidate gaps. Click &quot;generate gaps&quot; above, or
            run <code className="readout text-[0.875rem]">rb-gaps-detect &lt;source_id&gt; --save</code>{" "}
            for a single paper.
          </p>
        )}

        <ul>
          {gaps.map((gap, i) => (
            <GapCard key={gap.id} gap={gap} index={i} onReviewed={handleReviewed} />
          ))}
        </ul>
      </div>
    </main>
  );
}

function GapCard({
  gap,
  index,
  onReviewed,
}: {
  gap: CandidateGap;
  index: number;
  onReviewed: (g: CandidateGap) => void;
}) {
  const [note, setNote] = useState(gap.review_note ?? "");
  const [ratings, setRatings] = useState<Partial<GapRatings>>({
    correctness_rating: gap.correctness_rating,
    relevance_rating: gap.relevance_rating,
    novelty_rating: gap.novelty_rating,
    evidence_support_rating: gap.evidence_support_rating,
    usefulness_rating: gap.usefulness_rating,
  });
  const [busy, setBusy] = useState<"approved" | "rejected" | null>(null);
  const [failed, setFailed] = useState(false);

  async function review(next: "approved" | "rejected") {
    setBusy(next);
    setFailed(false);
    try {
      const updated = await gapsApi.review(gap.id, next, note, ratings);
      onReviewed(updated);
    } catch {
      setFailed(true);
    } finally {
      setBusy(null);
    }
  }

  return (
    <li
      className="resolve border-t border-[var(--rule-soft)] py-6 first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <Link href={`/papers/${gap.seed_paper_id}`} className="eyebrow hover:text-[var(--ink)]">
          seed: {gap.seed_paper_title}
        </Link>
        <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
          {gap.contributing_paper_count} papers · threshold {gap.similarity_threshold.toFixed(2)} ·{" "}
          {gap.detection_method}
        </span>
      </div>

      {gap.gap_status && (
        <span
          className={`eyebrow mt-2 inline-block rounded-[2px] border px-2 py-0.5 text-[0.6875rem] ${
            gap.gap_status === "strong_gap"
              ? "border-[var(--ink)] text-[var(--ink)]"
              : gap.gap_status === "potential_gap"
                ? "border-[var(--rule)] text-[var(--ink-soft)]"
                : "border-[var(--rule-soft)] text-[var(--ink-faint)]"
          }`}
        >
          {GAP_STATUS_LABELS[gap.gap_status]}
        </span>
      )}

      <div className="mt-3">
        <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">candidate gap</span>
        <p className="mt-1 max-w-[68ch] font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
          {gap.observation}
        </p>
      </div>

      <div className="mt-1 text-[0.75rem] text-[var(--ink-faint)]">
        inference — not stated by any single author
        {gap.claim && (
          <>
            {" "}
            · confidence: <span className="readout">{gap.claim.confidence}</span>
          </>
        )}
      </div>

      {gap.resolution_note && (
        <p className="mt-2 max-w-[68ch] text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
          ⚠ {gap.resolution_note}
        </p>
      )}

      {gap.evidence.length > 0 && (
        <details className="mt-4">
          <summary className="eyebrow inline-flex cursor-pointer items-center gap-1.5 hover:text-[var(--ink)]">
            supporting pattern — recurring across {gap.contributing_paper_count} related papers ({gap.evidence.length}{" "}
            evidence)
            <InfoTooltip text="The actual passages this candidate was inferred from — one per contributing paper. Read these before approving or rejecting; the candidate gap above is drawn from this pattern, not invented separately from it." />
          </summary>
          <ul className="mt-2 space-y-3 border-l border-[var(--rule-soft)] pl-4">
            {gap.evidence.map((ev, i) => (
              <li key={i}>
                <div className="flex flex-wrap items-center gap-2">
                  <Link href={`/papers/${ev.paper_id}`} className="eyebrow hover:text-[var(--ink)]">
                    {ev.paper_title}
                  </Link>
                  <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                    {ev.claim_type}
                    {ev.validation_tier ? ` · ${ev.validation_tier}` : ""}
                    {ev.claim_role
                      ? ` · ${ev.claim_role}`
                      : gap.gap_status
                        ? " · excluded from corroboration"
                        : ""}
                  </span>
                </div>
                <p className="mt-1 max-w-[62ch] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                  &quot;{ev.text}&quot;
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}

      {gap.status === "pending" ? (
        <div className="mt-5 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            {RATING_DIMENSIONS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1.5">
                <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">{label}</span>
                <select
                  value={ratings[key] ?? ""}
                  onChange={(e) =>
                    setRatings((r) => ({ ...r, [key]: e.target.value === "" ? null : Number(e.target.value) }))
                  }
                  className="readout border-b border-[var(--rule)] bg-transparent py-0.5 text-[0.8125rem] text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none"
                >
                  <option value="">—</option>
                  <option value="0">0 irrelevant</option>
                  <option value="1">1 weak</option>
                  <option value="2">2 plausible</option>
                  <option value="3">3 highly relevant</option>
                </select>
              </label>
            ))}
            <InfoTooltip text="Optional Sec 44 ratings on 0-3: irrelevant / weak / plausible / highly relevant. Saved alongside your approve/reject decision - leave any dimension blank to skip it." />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <input
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="optional note"
              className="min-w-[16rem] flex-1 border-b border-[var(--rule)] bg-transparent py-1 text-[0.875rem] text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:border-[var(--ink)] focus:outline-none"
            />
            <button
              onClick={() => review("approved")}
              disabled={busy !== null}
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
            >
              {busy === "approved" ? "approving…" : "approve"}
            </button>
            <button
              onClick={() => review("rejected")}
              disabled={busy !== null}
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--live)] hover:text-[var(--live)] disabled:opacity-50"
            >
              {busy === "rejected" ? "rejecting…" : "reject"}
            </button>
            <InfoTooltip text="Approve confirms this is a real, worth-tracking gap in the literature. Reject dismisses it as noise or a bad inference. Either choice removes it from the pending queue; the optional note is saved alongside your decision." />
            {failed && <span className="text-[0.75rem] text-[var(--live)]">save failed — try again</span>}
          </div>
        </div>
      ) : (
        <div className="mt-4 text-[0.8125rem] text-[var(--ink-soft)]">
          <span className="eyebrow">{gap.status}</span>
          {gap.review_note && <span className="ml-2">— {gap.review_note}</span>}
          {RATING_DIMENSIONS.some(({ key }) => gap[key] !== null) && (
            <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[0.75rem] text-[var(--ink-faint)]">
              {RATING_DIMENSIONS.filter(({ key }) => gap[key] !== null).map(({ key, label }) => (
                <span key={key}>
                  {label}: <span className="readout">{gap[key]}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </li>
  );
}
