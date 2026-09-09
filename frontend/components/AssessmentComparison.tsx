"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { assessmentApi, type ResearchAssessment } from "@/lib/assessmentApi";
import { Nav } from "@/components/Nav";
import { SkeletonReport } from "@/components/Skeleton";

type AssessmentComparisonProps = {
  ids: string[];
};

function displayValue(value: string | null | undefined): string {
  return value ? value.replaceAll("_", " ") : "not assessed";
}

function evidenceCount(assessment: ResearchAssessment): number {
  return assessment.evidence.length;
}

function paperCount(assessment: ResearchAssessment): number {
  return assessment.retrieved_paper_ids.length;
}

function sharedCount(
  assessments: ResearchAssessment[],
  getIds: (assessment: ResearchAssessment) => string[],
): number {
  if (assessments.length === 0) return 0;
  const [first, ...rest] = assessments.map(
    (assessment) => new Set(getIds(assessment)),
  );
  return [...first].filter((id) => rest.every((ids) => ids.has(id))).length;
}

export function AssessmentComparison({ ids }: AssessmentComparisonProps) {
  const [assessments, setAssessments] = useState<ResearchAssessment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Resetting fetch state for a new selection is intentional: it prevents
    // the previous comparison from remaining visible while new reports load.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);

    if (ids.length < 2) {
      setAssessments([]);
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }

    Promise.all(ids.map((id) => assessmentApi.get(id)))
      .then((items) => {
        if (!cancelled) setAssessments(items);
      })
      .catch(() => {
        if (!cancelled)
          setError("One or more selected assessments could not be loaded.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [ids]);

  const sharedPapers = sharedCount(
    assessments,
    (assessment) => assessment.retrieved_paper_ids,
  );
  const sharedEvidencePapers = sharedCount(assessments, (assessment) =>
    assessment.evidence.map((item) => item.paper_id),
  );

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <div className="flex flex-wrap items-baseline justify-between gap-4">
          <div>
            <span className="eyebrow">assessment comparison</span>
            <h1 className="display mt-3 max-w-[24ch] text-[2rem] leading-tight sm:text-[2.75rem]">
              Compare research decisions side by side.
            </h1>
          </div>
          <Link
            href="/assessments"
            className="eyebrow text-[var(--ink-soft)] hover:text-[var(--ink)]"
          >
            ← back to assessments
          </Link>
        </div>

        {ids.length < 2 && (
          <p className="mt-10 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
            Choose at least two completed assessments from the assessment list
            to compare them.
          </p>
        )}

        {loading && <SkeletonReport />}
        {error && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            {error}
          </p>
        )}

        {!loading && !error && assessments.length >= 2 && (
          <>
            <div className="mt-8 grid grid-cols-2 gap-px border-y border-[var(--rule)] bg-[var(--rule)] sm:grid-cols-4">
              <SummaryStat
                label="assessments"
                value={String(assessments.length)}
              />
              <SummaryStat
                label="shared retrieved papers"
                value={String(sharedPapers)}
              />
              <SummaryStat
                label="shared evidence papers"
                value={String(sharedEvidencePapers)}
              />
              <SummaryStat
                label="reviewed"
                value={`${assessments.filter((assessment) => assessment.human_reviewed).length}/${assessments.length}`}
              />
            </div>

            <div className="mt-10 overflow-x-auto border-y border-[var(--rule)]">
              <div
                className="min-w-[54rem]"
                style={{
                  display: "grid",
                  gridTemplateColumns: `minmax(10rem, 0.7fr) repeat(${assessments.length}, minmax(16rem, 1fr))`,
                }}
              >
                <div className="border-b border-[var(--rule)] p-4" />
                {assessments.map((assessment) => (
                  <div
                    key={assessment.id}
                    className="border-b border-l border-[var(--rule)] p-4"
                  >
                    <span className="eyebrow">
                      {assessment.research_input.input_type === "document"
                        ? "uploaded document"
                        : "research idea"}
                    </span>
                    <Link
                      href={`/assessments/${assessment.id}`}
                      className="mt-2 block text-[0.9375rem] leading-relaxed text-[var(--ink)] hover:underline"
                    >
                      {assessment.research_input.title ||
                        assessment.research_input.raw_text.slice(0, 140)}
                    </Link>
                    <span className="mt-2 block text-[0.6875rem] text-[var(--ink-faint)]">
                      {assessment.created_at
                        ? new Date(assessment.created_at).toLocaleString()
                        : "date unavailable"}
                    </span>
                  </div>
                ))}

                <ComparisonLabel label="recommendation" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-recommendation`}
                    value={displayValue(assessment.recommendation)}
                    emphasis
                  />
                ))}

                <ComparisonLabel label="confidence" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-confidence`}
                    value={displayValue(assessment.confidence)}
                  />
                ))}

                <ComparisonLabel label="novelty" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-novelty`}
                    value={displayValue(assessment.novelty_level)}
                  />
                ))}

                <ComparisonLabel label="technical feasibility" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-feasibility`}
                    value={displayValue(assessment.technical_feasibility_level)}
                  />
                ))}

                <ComparisonLabel label="retrieved papers" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-papers`}
                    value={String(paperCount(assessment))}
                  />
                ))}

                <ComparisonLabel label="evidence passages" />
                {assessments.map((assessment) => (
                  <ComparisonValue
                    key={`${assessment.id}-evidence`}
                    value={String(evidenceCount(assessment))}
                  />
                ))}

                <ComparisonLabel label="research gap" />
                {assessments.map((assessment) => (
                  <ComparisonText
                    key={`${assessment.id}-gap`}
                    text={assessment.research_gap_text}
                  />
                ))}

                <ComparisonLabel label="applications" />
                {assessments.map((assessment) => (
                  <ComparisonText
                    key={`${assessment.id}-applications`}
                    text={
                      assessment.potential_applications
                        ?.map((item) => item.application)
                        .join("\n") ?? null
                    }
                  />
                ))}

                <ComparisonLabel label="risks and limitations" />
                {assessments.map((assessment) => (
                  <ComparisonText
                    key={`${assessment.id}-risks`}
                    text={assessment.risks_and_limitations}
                  />
                ))}
              </div>
            </div>

            <p className="mt-5 max-w-[70ch] text-[0.75rem] leading-relaxed text-[var(--ink-faint)]">
              Shared-paper counts use the retrieved literature IDs. Evidence
              counts use only the passages attached to each report, so a paper
              can be retrieved without contributing evidence to a conclusion.
            </p>
          </>
        )}
      </div>
    </main>
  );
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[var(--panel)] px-4 py-4">
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        {label}
      </span>
      <span className="mt-1 block readout text-[1.1rem] text-[var(--ink)]">
        {value}
      </span>
    </div>
  );
}

function ComparisonLabel({ label }: { label: string }) {
  return (
    <div className="border-b border-[var(--rule-soft)] p-4 text-[0.75rem] text-[var(--ink-faint)]">
      {label}
    </div>
  );
}

function ComparisonValue({
  value,
  emphasis = false,
}: {
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className="border-b border-l border-[var(--rule-soft)] p-4 text-[0.875rem] text-[var(--ink)]">
      <span className={emphasis ? "font-medium" : ""}>{value}</span>
    </div>
  );
}

function ComparisonText({ text }: { text: string | null }) {
  return (
    <div className="min-h-20 whitespace-pre-line border-b border-l border-[var(--rule-soft)] p-4 text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
      {text || "not assessed"}
    </div>
  );
}
