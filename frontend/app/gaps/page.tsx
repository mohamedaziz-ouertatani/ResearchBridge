"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { gapsApi, type CandidateGap, type GapStatusFilter } from "@/lib/gapsApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";

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
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
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

        {loading && <p className="eyebrow py-16">loading…</p>}
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
  const [busy, setBusy] = useState<"approved" | "rejected" | null>(null);
  const [failed, setFailed] = useState(false);

  async function review(next: "approved" | "rejected") {
    setBusy(next);
    setFailed(false);
    try {
      const updated = await gapsApi.review(gap.id, next, note);
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

      <p className="mt-3 max-w-[68ch] font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
        {gap.observation}
      </p>

      <div className="mt-2 text-[0.75rem] text-[var(--ink-faint)]">
        {gap.gap_type} — a synthesized observation, not a stated fact
      </div>

      {gap.evidence.length > 0 && (
        <details className="mt-4">
          <summary className="eyebrow inline-flex cursor-pointer items-center gap-1.5 hover:text-[var(--ink)]">
            evidence ({gap.evidence.length})
            <InfoTooltip text="The actual passages this candidate was inferred from — one per contributing paper. Read these before approving or rejecting; the observation above is a synthesis of them, not a quote itself." />
          </summary>
          <ul className="mt-2 space-y-3 border-l border-[var(--rule-soft)] pl-4">
            {gap.evidence.map((ev, i) => (
              <li key={i}>
                <Link href={`/papers/${ev.paper_id}`} className="eyebrow hover:text-[var(--ink)]">
                  {ev.paper_title}
                </Link>
                <p className="mt-1 max-w-[62ch] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                  &quot;{ev.text}&quot;
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}

      {gap.status === "pending" ? (
        <div className="mt-5 flex flex-wrap items-center gap-3">
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
      ) : (
        <div className="mt-4 text-[0.8125rem] text-[var(--ink-soft)]">
          <span className="eyebrow">{gap.status}</span>
          {gap.review_note && <span className="ml-2">— {gap.review_note}</span>}
        </div>
      )}
    </li>
  );
}
