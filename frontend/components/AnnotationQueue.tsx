"use client";

import Link from "next/link";
import type { AnnotationSummary } from "@/lib/benchmarkApi";

/*
  The queue is a worklist, not a nav menu: forty papers is a long haul, and
  the thing that keeps it going is seeing exactly how far in you are and
  which one is next. Each row carries its own fill count.
*/
export function AnnotationQueue({
  papers,
  activeId,
}: {
  papers: AnnotationSummary[];
  activeId: string;
}) {
  return (
    <nav aria-label="Benchmark papers" className="flex flex-col">
      {papers.map((paper) => {
        const isActive = paper.source_id === activeId;
        return (
          <Link
            key={paper.source_id}
            href={`/annotate/${paper.source_id}`}
            aria-current={isActive ? "page" : undefined}
            className={`border-l-2 px-3 py-2.5 transition-colors ${
              isActive
                ? "border-[var(--near)] bg-[var(--panel)]"
                : "border-transparent hover:bg-[var(--panel)]"
            }`}
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                {paper.source_id}
              </span>
              <span
                className="readout text-[0.625rem]"
                style={{ color: paper.is_complete ? "var(--near)" : "var(--ink-faint)" }}
              >
                {paper.is_complete ? "done" : `${paper.filled}/${paper.total}`}
              </span>
            </div>
            <p className="mt-0.5 line-clamp-2 text-[0.8125rem] leading-snug text-[var(--ink)]">
              {paper.title ?? paper.source_id}
            </p>
            {!paper.has_fulltext && (
              <span className="eyebrow mt-1 inline-block text-[var(--live)]">no full text</span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}
