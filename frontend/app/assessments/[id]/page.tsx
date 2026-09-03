"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { assessmentApi, type ResearchAssessment } from "@/lib/assessmentApi";
import { AssessmentReport } from "@/components/AssessmentReport";
import { SimilarityGraph } from "@/components/SimilarityGraph";
import { SkeletonReport } from "@/components/Skeleton";

export default function AssessmentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const [assessment, setAssessment] = useState<ResearchAssessment | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Clearing the stale assessment/error before fetching by `id` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAssessment(null);
    setError(null);
    assessmentApi
      .get(id)
      .then(setAssessment)
      .catch(() => setError("That assessment isn't available."));
  }, [id]);

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
        <span className="eyebrow">research assessment</span>
      </header>

      {error && <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">{error}</p>}

      {!assessment && !error && <SkeletonReport />}

      {assessment && (
        <div className="pt-12">
          <AssessmentReport assessment={assessment} />
          <SimilarityGraph assessmentId={id} />
        </div>
      )}
    </main>
  );
}
