"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  assessmentApi,
  type AssessmentSort,
  type AssessmentSummary,
  type CategoricalLevel,
  type NoveltyLevel,
  type ReviewFilter,
} from "@/lib/assessmentApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";
import { SkeletonRows } from "@/components/Skeleton";

const FILTERS: ReviewFilter[] = ["needs_review", "reviewed", "all"];
// feasibility never takes "insufficient_evidence" (see novelty.py) - kept
// as its own list so that value only ever appears in the novelty dropdown.
const NOVELTY_LEVEL_OPTIONS: NoveltyLevel[] = ["high", "medium", "low", "insufficient_evidence", "not_assessed"];
const FEASIBILITY_LEVEL_OPTIONS: CategoricalLevel[] = ["high", "medium", "low", "not_assessed"];

const SELECT_CLASS =
  "eyebrow rounded-[2px] border border-[var(--rule)] bg-transparent px-2 py-1 text-[0.6875rem] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none";

export default function AssessmentDashboard() {
  const [review, setReview] = useState<ReviewFilter>("needs_review");
  const [sort, setSort] = useState<AssessmentSort>("newest");
  const [novelty, setNovelty] = useState<NoveltyLevel | "">("");
  const [feasibility, setFeasibility] = useState<CategoricalLevel | "">("");
  const [assessments, setAssessments] = useState<AssessmentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Resetting loading/error before a fetch keyed on the filters below is
    // intentional - not the accidental-derived-state case this rule
    // otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    assessmentApi
      .list(review, { sort, novelty: novelty || undefined, feasibility: feasibility || undefined })
      .then((page) => setAssessments(page.items))
      .catch(() => setError("Couldn't load assessments."))
      .finally(() => setLoading(false));
  }, [review, sort, novelty, feasibility]);

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">assessments</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every idea or upload that has been checked against the literature. Re-running an
          assessment keeps its older versions in that assessment&apos;s own history — this list
          always shows only the latest.
        </p>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every assessment is created from the{" "}
          <Link href="/" className="underline hover:text-[var(--ink)]">
            home page
          </Link>
          , either by describing an idea or uploading a paper. This page is where you come back to
          find one again, and where any that need a closer human look are flagged for review.
        </p>

        <div className="mt-6 flex items-center gap-1 border-b border-[var(--rule)]">
          <InfoTooltip text={'"Needs review" is every assessment nobody has marked reviewed yet. "Reviewed" is everything a person has already looked at and confirmed. "All" ignores review status entirely.'} />
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

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">sort</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as AssessmentSort)}
              className={SELECT_CLASS}
            >
              <option value="newest">newest</option>
              <option value="priority">priority</option>
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">novelty</span>
            <select
              value={novelty}
              onChange={(e) => setNovelty(e.target.value as NoveltyLevel | "")}
              className={SELECT_CLASS}
            >
              <option value="">any</option>
              {NOVELTY_LEVEL_OPTIONS.map((level) => (
                <option key={level} value={level}>
                  {level.replace("_", " ")}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">feasibility</span>
            <select
              value={feasibility}
              onChange={(e) => setFeasibility(e.target.value as CategoricalLevel | "")}
              className={SELECT_CLASS}
            >
              <option value="">any</option>
              {FEASIBILITY_LEVEL_OPTIONS.map((level) => (
                <option key={level} value={level}>
                  {level.replace("_", " ")}
                </option>
              ))}
            </select>
          </label>

          <InfoTooltip text={'"Priority" sorts by the recommendation the pipeline already computed (HIGH → MEDIUM → LOW → REQUIRES HUMAN REVIEW → INSUFFICIENT EVIDENCE), not a new invented score. Novelty/feasibility filters narrow the list to assessments at that categorical level.'} />
        </div>

        {loading && <SkeletonRows count={5} />}
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
          <span className="readout text-[var(--ink-faint)]">
            feasibility: {assessment.technical_feasibility_level.replace("_", " ")}
          </span>
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
