import Link from "next/link";
import type { PaperSummary } from "@/lib/api";
import { ProximityGauge } from "./ProximityGauge";

function year(date: string | null) {
  return date ? date.slice(0, 4) : "—";
}

function authorLine(authors: string[]) {
  if (authors.length === 0) return "Unattributed";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} +${authors.length - 3}`;
}

export function PaperRow({
  paper,
  distance,
  index = 0,
}: {
  paper: PaperSummary;
  distance?: number;
  index?: number;
}) {
  return (
    <li
      className="resolve border-t border-[var(--rule-soft)] first:border-t-0"
      style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
    >
      <Link
        href={`/papers/${paper.id}`}
        className="group grid gap-x-6 gap-y-2 px-1 py-5 transition-colors hover:bg-[var(--panel)] sm:grid-cols-[minmax(0,1fr)_auto]"
      >
        <div className="min-w-0">
          <div className="mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">{paper.source_id}</span>
            {paper.primary_category && (
              <span className="readout rounded-[2px] bg-[var(--field)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]">
                {paper.primary_category}
              </span>
            )}
            {paper.excluded_at && (
              <span className="readout rounded-[2px] border border-[var(--live)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--live)]">
                excluded
              </span>
            )}
            <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">{year(paper.publication_date)}</span>
          </div>

          <h3 className="font-[family-name:var(--type-text)] text-[1.0625rem] leading-snug text-[var(--ink)] underline-offset-4 group-hover:underline">
            {paper.title}
          </h3>

          <p className="mt-1.5 truncate font-[family-name:var(--type-display)] text-[0.8125rem] text-[var(--ink-soft)]">
            {authorLine(paper.authors)}
          </p>
        </div>

        {distance !== undefined && (
          <div className="sm:w-[240px] sm:justify-self-end sm:pt-6">
            <ProximityGauge distance={distance} />
          </div>
        )}
      </Link>
    </li>
  );
}
