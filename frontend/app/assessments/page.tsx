"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { assessmentApi, type AssessmentSummary, type ReviewFilter } from "@/lib/assessmentApi";

const FILTERS: ReviewFilter[] = ["needs_review", "reviewed", "all"];

export default function AssessmentDashboard() {
  const [review, setReview] = useState<ReviewFilter>("needs_review");
  const [assessments, setAssessments] = useState<AssessmentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    assessmentApi
      .list(review)
      .then((page) => setAssessments(page.items))
      .catch(() => setError("Couldn't load assessments."))
      .finally(() => setLoading(false));
  }, [review]);

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
      </header>

      <div className="pt-12">
        <span className="eyebrow">assessments</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every idea or upload that has been checked against the literature. Re-running an
          assessment keeps its older versions in that assessment&apos;s own history — this list
          always shows only the latest.
        </p>

        <div className="mt-6 flex gap-1 border-b border-[var(--rule)]">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setReview(f)}
              className={`eyebrow -mb-px border-b-2 px-3 py-2 ${
                review === f
                  ? "border-[var(--ink)] text-[var(--ink)]"
                  : "border-transparent text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
              }`}
            >
              {f.replace("_", " ")}
            </button>
          ))}
        </div>

        {loading && <p className="eyebrow py-16">loading…</p>}
        {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

        {!loading && !error && assessments.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            No {review === "all" ? "" : review.replace("_", " ")} assessments yet.
          </p>
        )}

        <ul>
          {assessments.map((assessment, i) => (
            <AssessmentRow
              key={assessment.id}
              assessment={assessment}
              index={i}
              onDeleted={() => setAssessments((rows) => rows.filter((r) => r.id !== assessment.id))}
            />
          ))}
        </ul>
      </div>
    </main>
  );
}

function AssessmentRow({
  assessment,
  index,
  onDeleted,
}: {
  assessment: AssessmentSummary;
  index: number;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirmDelete() {
    setDeleting(true);
    setError(null);
    try {
      await assessmentApi.remove(assessment.id);
      onDeleted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't delete this assessment.");
      setDeleting(false);
    }
  }

  return (
    <li
      className="resolve flex items-start gap-4 border-t border-[var(--rule-soft)] py-6 first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <Link href={`/assessments/${assessment.id}`} className="min-w-0 flex-1 hover:opacity-80">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="eyebrow">{assessment.input_type === "document" ? "uploaded document" : "research idea"}</span>
          <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
            {new Date(assessment.created_at).toLocaleString()}
          </span>
        </div>

        <p className="mt-2 max-w-[68ch] font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
          {assessment.input_preview}
        </p>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
          <span>{assessment.recommendation ?? "Not assessed"}</span>
          <span className="readout text-[var(--ink-faint)]">novelty: {assessment.novelty_level.replace("_", " ")}</span>
          {assessment.confidence && (
            <span className="readout text-[var(--ink-faint)]">confidence: {assessment.confidence}</span>
          )}
          <span className="eyebrow">{assessment.human_reviewed ? "✓ reviewed" : "needs review"}</span>
        </div>

        {error && <p className="mt-2 text-[0.75rem] text-[var(--live)]">{error}</p>}
      </Link>

      {/* Sibling of the Link, not nested inside it, so a click here never
       * triggers navigation - no preventDefault/stopPropagation needed. A
       * normal-flow flex-none column (not absolutely positioned) so it
       * always reserves its own space instead of overlapping the
       * timestamp above whenever the row's content wraps. */}
      <div className="flex flex-none items-baseline gap-2 pt-px">
        {!confirming ? (
          <button
            type="button"
            onClick={() => setConfirming(true)}
            aria-label="Delete this assessment"
            className="eyebrow text-[0.6875rem] text-[var(--ink-faint)] hover:text-[var(--live)]"
          >
            delete
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={() => void confirmDelete()}
              disabled={deleting}
              className="eyebrow text-[0.6875rem] text-[var(--live)] hover:underline disabled:opacity-50"
            >
              {deleting ? "deleting…" : "confirm?"}
            </button>
            <button
              type="button"
              onClick={() => setConfirming(false)}
              disabled={deleting}
              className="eyebrow text-[0.6875rem] text-[var(--ink-faint)] hover:text-[var(--ink)] disabled:opacity-50"
            >
              cancel
            </button>
          </>
        )}
      </div>
    </li>
  );
}
