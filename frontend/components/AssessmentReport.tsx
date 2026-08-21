import Link from "next/link";
import { useState } from "react";
import { assessmentApi, type AssessmentEvidence, type EvidenceRole, type ResearchAssessment } from "@/lib/assessmentApi";

/*
  The assessment report, read as an instrument readout sheet.

  Its signature is the evidence gutter: every field carries a mono readout of
  how many real quoted passages back it, and expands to show them. A field
  with nothing behind it reads "—" and says so in words. That is the whole
  premise of this project made visible - an assessment is a lightweight
  summary, never its own source of truth (blueprint Sec 15/17), so a reader
  can see at a glance how much of the report the literature actually supports
  rather than taking the prose on trust.

  Categorical levels (novelty, feasibility, confidence) are deliberately NOT
  drawn on the --near/--far teal ramp. That ramp encodes measured cosine
  distance and nothing else (see globals.css), and these levels are rule-based
  judgements, not measurements - the same reasoning that keeps extraction
  confidence uncolored in ExtractedClaims.tsx.
*/

/** The seven fields that can be backed by retrieved literature. */
const GRADEABLE_ROLES: EvidenceRole[] = [
  "comparison",
  "novelty",
  "research_gap",
  "application",
  "feasibility",
  "risk",
  "opportunity",
];

/** Extraction field keys, as the reader should see them (mirrors Sec 25/28 order). */
const CLAIM_LABELS: Record<string, string> = {
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

function groupByRole(evidence: AssessmentEvidence[]) {
  const byRole = new Map<EvidenceRole, AssessmentEvidence[]>();
  for (const item of evidence) {
    const existing = byRole.get(item.role);
    if (existing) existing.push(item);
    else byRole.set(item.role, [item]);
  }
  return byRole;
}

export function AssessmentReport({ assessment }: { assessment: ResearchAssessment }) {
  const byRole = groupByRole(assessment.evidence);
  const groundedCount = GRADEABLE_ROLES.filter((role) => (byRole.get(role)?.length ?? 0) > 0).length;

  const contributingPapers = new Map<string, string>();
  for (const item of assessment.evidence) contributingPapers.set(item.paper_id, item.paper_title);

  return (
    <article className="resolve">
      <Verdict assessment={assessment} grounded={groundedCount} total={GRADEABLE_ROLES.length} />

      <Field label="input" gradeable={false}>
        <p className="font-[family-name:var(--type-text)] text-[1.0625rem] leading-[1.7]">
          {assessment.research_input.raw_text.length > 600
            ? `${assessment.research_input.raw_text.slice(0, 600)}…`
            : assessment.research_input.raw_text}
        </p>
        <p className="eyebrow mt-3">
          {assessment.research_input.input_type === "document" ? "uploaded document" : "research idea"}
        </p>
        {assessment.research_input.matched_paper_id && (
          <p className="mt-2 text-[0.8125rem] text-[var(--ink-soft)]">
            matched to an existing corpus paper —{" "}
            <Link
              href={`/papers/${assessment.research_input.matched_paper_id}`}
              className="underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
            >
              view it
            </Link>
          </p>
        )}
      </Field>

      <Field label="related research" gradeable={false}>
        {contributingPapers.size === 0 ? (
          <Unassessed reason="No retrieved paper contributed usable extracted claims." />
        ) : (
          <>
            <ul className="space-y-1">
              {[...contributingPapers].map(([paperId, title]) => (
                <li key={paperId}>
                  <Link
                    href={`/papers/${paperId}`}
                    className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
                  >
                    {title}
                  </Link>
                </li>
              ))}
            </ul>
            <p className="readout mt-3 text-[0.6875rem] text-[var(--ink-faint)]">
              {assessment.retrieved_paper_ids.length} retrieved · {contributingPapers.size} contributed evidence
            </p>
          </>
        )}
      </Field>

      <Field label="existing solutions" evidence={byRole.get("comparison")}>
        {assessment.comparison_summary ? (
          <ComparisonSummary text={assessment.comparison_summary} />
        ) : (
          <Unassessed reason="No retrieved paper had extracted claims to compare against." />
        )}
      </Field>

      <Field label="novelty assessment" evidence={byRole.get("novelty")} level={assessment.novelty_level}>
        {assessment.novelty_reasoning ? (
          <Prose text={assessment.novelty_reasoning} />
        ) : (
          <Unassessed reason="Not enough evidence to judge novelty." />
        )}
      </Field>

      <Field label="research gap" evidence={byRole.get("research_gap")}>
        {assessment.research_gap_text ? (
          <>
            <Prose text={assessment.research_gap_text} />
            {assessment.research_gap_source && (
              <p className="eyebrow mt-3">
                {assessment.research_gap_source === "reused_candidate_gap"
                  ? "reused a reviewed candidate gap"
                  : "found for this input"}
              </p>
            )}
          </>
        ) : (
          <Unassessed reason="No gap was found in the retrieved literature for this input." />
        )}
      </Field>

      <Field label="potential applications" evidence={byRole.get("application")}>
        {assessment.potential_applications?.length ? (
          <ul className="space-y-3">
            {assessment.potential_applications.map((app, i) => (
              <li key={i}>
                <p className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed">
                  {app.application}
                </p>
                <Link href={`/papers/${app.paper_id}`} className="eyebrow mt-1 inline-block hover:text-[var(--ink)]">
                  {app.source_paper}
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <Unassessed reason="No retrieved paper stated an application." />
        )}
      </Field>

      <Field label="product / technology opportunities" evidence={byRole.get("opportunity")}>
        <Unassessed reason="Not generated. Naming a product opportunity means inventing a claim the literature does not make, so this is left to a human reviewer." />
      </Field>

      <Field
        label="technical feasibility"
        evidence={byRole.get("feasibility")}
        level={assessment.technical_feasibility_level}
      >
        {assessment.technical_feasibility_reasoning ? (
          <Prose text={assessment.technical_feasibility_reasoning} />
        ) : (
          <Unassessed reason="Nothing close enough to ground a feasibility judgement." />
        )}
      </Field>

      <Field label="risks / limitations" evidence={byRole.get("risk")}>
        {assessment.risks_and_limitations ? (
          <Preformatted text={assessment.risks_and_limitations} />
        ) : (
          <Unassessed reason="No retrieved paper stated a limitation." />
        )}
      </Field>

      <Field label="external validation needed" gradeable={false}>
        <Prose text={assessment.external_validation_needed} />
      </Field>
    </article>
  );
}

function Verdict({
  assessment,
  grounded,
  total,
}: {
  assessment: ResearchAssessment;
  grounded: number;
  total: number;
}) {
  const [reviewed, setReviewed] = useState(assessment.human_reviewed);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  async function toggleReviewed() {
    setBusy(true);
    setFailed(false);
    try {
      const updated = await assessmentApi.review(assessment.id, !reviewed);
      setReviewed(updated.human_reviewed);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <header className="border-b border-[var(--rule)] pb-8">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div>
          <span className="eyebrow">recommendation</span>
          <p className="display mt-3 text-[clamp(1.75rem,4.5vw,2.75rem)]">
            {assessment.recommendation ?? "Not assessed"}
          </p>
        </div>

        <button
          onClick={toggleReviewed}
          disabled={busy}
          className={`eyebrow shrink-0 rounded-[2px] border px-3 py-1.5 disabled:opacity-50 ${
            reviewed
              ? "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
              : "border-[var(--rule)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
          }`}
        >
          {busy ? "saving…" : reviewed ? "✓ reviewed · un-mark" : "mark reviewed"}
        </button>
      </div>

      {failed && <p className="mt-2 text-[0.75rem] text-[var(--live)]">save failed — try again</p>}

      <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-3">
        <div>
          <dt className="eyebrow">confidence</dt>
          <dd className="readout mt-1 text-[0.9375rem]">{assessment.confidence ?? "—"}</dd>
        </div>
        <div>
          <dt className="eyebrow">grounded fields</dt>
          <dd className="readout mt-1 text-[0.9375rem]">
            {grounded} / {total}
          </dd>
        </div>
      </dl>

      <p className="mt-5 max-w-[58ch] text-[0.875rem] leading-relaxed text-[var(--ink-faint)]">
        Confidence counts how many signals could be assessed, not how likely this reading is to be
        right. Fields with no supporting passage are left unassessed rather than filled in.
      </p>
    </header>
  );
}

function Field({
  label,
  level,
  evidence,
  gradeable = true,
  children,
}: {
  label: string;
  level?: string;
  evidence?: AssessmentEvidence[];
  /** False for fields the literature cannot back either way - the reader's own
      input, the list of papers retrieved, the external-validation disclaimer.
      Their gutter stays blank so "—" keeps one precise meaning: this reading
      could have been grounded and wasn't. */
  gradeable?: boolean;
  children: React.ReactNode;
}) {
  const count = evidence?.length ?? 0;

  return (
    <section className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-x-4 border-b border-[var(--rule-soft)] py-7 sm:grid-cols-[3.5rem_minmax(0,1fr)] sm:gap-x-6">
      {/* the evidence gutter: how many real passages back this reading */}
      <div className="pt-0.5">
        {gradeable && (
          <span
            className="readout text-[0.8125rem] text-[var(--ink-faint)] tabular-nums"
            title={count > 0 ? `${count} supporting passage${count === 1 ? "" : "s"}` : "no supporting passage"}
          >
            {count > 0 ? count : "—"}
          </span>
        )}
      </div>

      <div>
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <span className="eyebrow">{label}</span>
          {level && <span className="readout text-[0.75rem] text-[var(--ink-soft)]">{level.replace("_", " ")}</span>}
        </div>

        <div className="mt-3">{children}</div>

        {count > 0 && evidence && (
          <details className="mt-4">
            <summary className="eyebrow cursor-pointer hover:text-[var(--ink)]">
              supporting passages ({count})
            </summary>
            <ul className="mt-3 space-y-3 border-l border-[var(--rule-soft)] pl-4">
              {evidence.map((item, i) => (
                <li key={i}>
                  <p className="max-w-[64ch] font-[family-name:var(--type-text)] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                    “{item.text}”
                  </p>
                  <Link href={`/papers/${item.paper_id}`} className="eyebrow mt-1 inline-block hover:text-[var(--ink)]">
                    {item.paper_title}
                    {item.section ? ` · ${item.section}` : ""}
                  </Link>
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </section>
  );
}

function Prose({ text }: { text: string }) {
  return (
    <p className="max-w-[64ch] font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.7]">{text}</p>
  );
}

/*
  comparison_summary arrives as blocks the pipeline assembled:

      Paper Title
      - method: <claim text>
      - limitations: <claim text>

  Rendered rather than printed raw, so the reader sees "Method", not the
  extraction pipeline's own field key.
*/
function ComparisonSummary({ text }: { text: string }) {
  const blocks = text.split("\n\n").filter(Boolean);

  return (
    <div className="max-w-[64ch] space-y-6">
      {blocks.map((block, blockIndex) => {
        const [title, ...claimLines] = block.split("\n");
        return (
          <div key={blockIndex}>
            <p className="font-[family-name:var(--type-text)] text-[0.9375rem] font-semibold leading-snug">
              {title}
            </p>
            <dl className="mt-2 space-y-1.5">
              {claimLines.map((line, lineIndex) => {
                const match = /^-\s*([a-z_]+):\s*(.*)$/.exec(line);
                if (!match) return null;
                const [, key, claimText] = match;
                return (
                  <div key={lineIndex} className="grid grid-cols-1 gap-x-4 sm:grid-cols-[9rem_minmax(0,1fr)]">
                    <dt className="eyebrow sm:pt-1">{CLAIM_LABELS[key] ?? key.replace(/_/g, " ")}</dt>
                    <dd className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
                      {claimText}
                    </dd>
                  </div>
                );
              })}
            </dl>
          </div>
        );
      })}
    </div>
  );
}

/** Content the pipeline assembled line-by-line (one line per source paper). */
function Preformatted({ text }: { text: string }) {
  return (
    <div className="max-w-[64ch] space-y-2">
      {text.split("\n").filter(Boolean).map((line, i) => (
        <p key={i} className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed">
          {line}
        </p>
      ))}
    </div>
  );
}

function Unassessed({ reason }: { reason: string }) {
  return <p className="max-w-[58ch] text-[0.875rem] leading-relaxed text-[var(--ink-faint)]">{reason}</p>;
}
