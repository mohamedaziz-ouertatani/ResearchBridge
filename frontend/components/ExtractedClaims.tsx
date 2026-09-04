import type { ExtractedClaim } from "@/lib/api";

/*
  Deliberately does NOT reuse the proximity gauge's teal-to-gray scale for
  confidence. That scale means "trustworthy -> weak", and confidence here
  doesn't track measured accuracy that way: "problem" claims are labeled
  "low" whenever the extractor falls back to an abstract's opening
  sentence, yet problem is the single most reliable field measured (0.81
  F1) - well above fields labeled "medium" (e.g. dataset, 0.15 F1). Coloring
  by confidence would visually assert a trust signal the data doesn't
  support. A plain text label states what it actually is: the extractor's
  own self-report, not a validated score.
*/

const FIELD_LABELS: Record<string, string> = {
  problem: "Problem",
  research_question: "Research question",
  method: "Method",
  dataset: "Dataset",
  main_contribution: "Main contribution",
  results: "Results",
  limitations: "Limitations",
  research_gap: "Research gap",
  applications: "Applications",
};

// "abstract" is the common case (most papers have no fetched full text
// yet) - only call out a section when it's a named full-text section, so
// the tag reads as "this one's different" rather than clutter on every row.
function sectionLabel(section: string | null): string | null {
  if (!section || section === "abstract") return null;
  return section.replace(/_/g, " ");
}

export function ExtractedClaims({ claims }: { claims: ExtractedClaim[] }) {
  if (claims.length === 0) return null;

  return (
    <section className="mt-16">
      <div className="border-b border-[var(--rule)] pb-3">
        <span className="eyebrow">extracted</span>
        <p className="mt-2 max-w-[60ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
          Pulled automatically from the abstract - or, when available, the paper&apos;s full text - by a
          baseline extractor, not written or checked by a person. Treat these as a starting point, not a
          citation.
        </p>
      </div>

      <ul className="mt-2">
        {claims.map((claim, i) => {
          const section = sectionLabel(claim.section);
          return (
            <li
              key={claim.claim_type}
              className="resolve border-t border-[var(--rule-soft)] py-4 first:border-t-0"
              style={{ animationDelay: `${Math.min(i * 28, 280)}ms` }}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <span className="eyebrow">{FIELD_LABELS[claim.claim_type] ?? claim.claim_type}</span>
                <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                  {claim.confidence} confidence{section ? ` · from ${section}` : ""}
                </span>
              </div>
              <p className="mt-2 font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
                {claim.text}
              </p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
