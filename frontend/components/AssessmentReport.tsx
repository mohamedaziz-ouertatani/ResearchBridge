import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  assessmentApi,
  type AssessmentEvidence,
  type AssessmentHistoryItem,
  type EvidenceRole,
  type PotentialApplication,
  type PotentialOpportunity,
  type ResearchAssessment,
} from "@/lib/assessmentApi";
import { api, API_BASE } from "@/lib/api";

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

export function groupByRole(evidence: AssessmentEvidence[]) {
  const byRole = new Map<EvidenceRole, AssessmentEvidence[]>();
  for (const item of evidence) {
    const existing = byRole.get(item.role);
    if (existing) existing.push(item);
    else byRole.set(item.role, [item]);
  }
  return byRole;
}

export function AssessmentReport({
  assessment,
  onAssessmentUpdated,
}: {
  assessment: ResearchAssessment;
  /** Called with the full updated assessment after opportunity synthesis
   * persists new evidence links - lets a parent that owns assessment state
   * (see app/assessments/[id]/page.tsx) refresh the "grounded fields" tally
   * and evidence gutter immediately, instead of only Opportunities' own
   * local state changing. Optional: omitting it still updates the
   * opportunities section itself (Opportunities keeps its own state too),
   * just not the rest of the report until the next fetch. */
  onAssessmentUpdated?: (updated: ResearchAssessment) => void;
}) {
  const byRole = groupByRole(assessment.evidence);
  const groundedCount = GRADEABLE_ROLES.filter((role) => (byRole.get(role)?.length ?? 0) > 0).length;

  const contributingPapers = new Map<string, string>();
  for (const item of assessment.evidence) contributingPapers.set(item.paper_id, item.paper_title);

  return (
    <article className="resolve">
      <Verdict assessment={assessment} grounded={groundedCount} total={GRADEABLE_ROLES.length} />

      <AssessmentHistory assessmentId={assessment.id} />

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

      <Field label="when this idea was discussed" gradeable={false}>
        <IdeaYearTrend evidence={assessment.evidence} />
      </Field>

      <Field label="existing solutions" evidence={byRole.get("comparison")}>
        {assessment.comparison_summary ? (
          <ComparisonSummary text={assessment.comparison_summary} />
        ) : (
          <Unassessed reason="No retrieved paper had extracted claims to compare against." />
        )}
      </Field>

      <Field label="corpus similarity / novelty signal" evidence={byRole.get("novelty")} level={assessment.novelty_level}>
        {assessment.novelty_reasoning ? (
          <Prose text={assessment.novelty_reasoning} />
        ) : (
          <Unassessed reason="Not enough evidence to judge corpus similarity." />
        )}
        <p className="mt-3 max-w-[58ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
          This reflects similarity to the papers retrieved from this corpus, not proof of global
          originality - the corpus is one CS/AI/ML slice, not the full scientific literature.
        </p>
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
          <Unassessed
            reason={
              assessment.research_gap_source === "no_relevant_evidence"
                ? "Not assessed - insufficient relevant evidence was retrieved for this input to investigate whether a research gap exists."
                : "No gap was found in the retrieved literature for this input."
            }
          />
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
          <Unassessed
            reason={
              assessment.potential_applications === null
                ? "No relevant paper was retrieved for this input, so applications could not be assessed."
                : "No retrieved paper stated an application."
            }
          />
        )}
      </Field>

      <Field label="product / technology opportunities" evidence={byRole.get("opportunity")}>
        <Opportunities
          assessmentId={assessment.id}
          applications={assessment.potential_applications}
          opportunities={assessment.potential_opportunities}
          onSynthesized={onAssessmentUpdated}
        />
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
        <p className="mt-3 max-w-[58ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
          This means documented technical grounding in the retrieved literature - a related method
          or dataset already exists on paper - not a prediction of whether this specific idea will
          succeed.
        </p>
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

      <Field label="recommendation reasoning" gradeable={false}>
        <Preformatted
          text={[
            `Novelty signal: ${assessment.novelty_level}`,
            assessment.novelty_reasoning ?? "",
            "",
            assessment.research_gap_text
              ? `Research gap evidence: ${assessment.research_gap_text}`
              : `Research gap evidence: ${
                  assessment.research_gap_source === "no_relevant_evidence"
                    ? "not assessed - insufficient relevant evidence"
                    : "none found - relevant literature was checked"
                }`,
            "",
            `Technical grounding: ${assessment.technical_feasibility_level}`,
            assessment.technical_feasibility_reasoning ?? "",
            "",
            `Recommendation: ${assessment.recommendation ?? "not assessed"}`,
            `Confidence: ${assessment.confidence ?? "not assessed"}`,
          ]
            .filter((line) => line !== undefined)
            .join("\n")}
        />
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
  const router = useRouter();
  const [reviewed, setReviewed] = useState(assessment.human_reviewed);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [rerunFailed, setRerunFailed] = useState(false);

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

  async function rerun() {
    setRerunning(true);
    setRerunFailed(false);
    try {
      const next = await assessmentApi.rerun(assessment.id);
      router.push(`/assessments/${next.id}`);
    } catch {
      setRerunFailed(true);
      setRerunning(false);
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

        <div className="flex shrink-0 flex-wrap gap-2">
          <a
            href={`${API_BASE}/api/assessments/${assessment.id}/export.docx`}
            className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)]"
          >
            export .docx
          </a>

          <a
            href={`${API_BASE}/api/assessments/${assessment.id}/export.pdf`}
            className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)]"
          >
            export .pdf
          </a>

          <button
            onClick={rerun}
            disabled={rerunning}
            className="eyebrow rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
          >
            {rerunning ? "re-running…" : "re-run assessment"}
          </button>

          <button
            onClick={toggleReviewed}
            disabled={busy}
            className={`eyebrow rounded-[2px] border px-3 py-1.5 disabled:opacity-50 ${
              reviewed
                ? "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
                : "border-[var(--rule)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
            }`}
          >
            {busy ? "saving…" : reviewed ? "✓ reviewed · un-mark" : "mark reviewed"}
          </button>
        </div>
      </div>

      {failed && <p className="mt-2 text-[0.75rem] text-[var(--live)]">save failed — try again</p>}
      {rerunFailed && <p className="mt-2 text-[0.75rem] text-[var(--live)]">re-run failed — try again</p>}

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

/** Sibling assessments for the same research_input (re-runs) - hidden entirely
    when there's only one, since a history of one isn't a history. */
function AssessmentHistory({ assessmentId }: { assessmentId: string }) {
  const [history, setHistory] = useState<AssessmentHistoryItem[] | null>(null);

  useEffect(() => {
    assessmentApi
      .history(assessmentId)
      .then(setHistory)
      .catch(() => setHistory(null));
  }, [assessmentId]);

  if (!history || history.length <= 1) return null;

  return (
    <section className="border-b border-[var(--rule-soft)] py-6">
      <span className="eyebrow">assessment history</span>
      <ul className="mt-3 space-y-2">
        {history.map((item) => (
          <li key={item.id} className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            {item.id === assessmentId ? (
              <span className="readout text-[0.8125rem]">viewing this one</span>
            ) : (
              <Link
                href={`/assessments/${item.id}`}
                className="readout text-[0.8125rem] underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
              >
                view
              </Link>
            )}
            <span className="text-[0.8125rem] text-[var(--ink-soft)]">
              {new Date(item.created_at).toLocaleString()} · {item.novelty_level.replace("_", " ")}
              {item.human_reviewed ? " · reviewed" : ""}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/*
  How many supporting passages trace back to a paper published in each year -
  a proxy for when the literature was mostly discussing this idea. Built from
  the evidence gutter's own paper_ids (not retrieved_paper_ids), so it only
  counts years the report actually grounds a claim in.

  Drawn as a single-series line rather than YearStrip's bar strip: a trend
  over a continuous year axis reads as change-over-time, so the gaps between
  discussed years (filled with zero, not skipped) are as meaningful as the
  peaks. One series needs no legend - the section title already names it -
  so this stays plain ink on the chart surface, no categorical palette.
*/
function IdeaYearTrend({ evidence }: { evidence: AssessmentEvidence[] }) {
  const [byYear, setByYear] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const paperIds = [...new Set(evidence.map((item) => item.paper_id))];

    if (paperIds.length === 0) {
      setByYear({});
      return;
    }

    Promise.all(
      paperIds.map((id) =>
        api
          .paper(id)
          .then((paper) => [id, paper.publication_date] as const)
          .catch(() => [id, null] as const),
      ),
    ).then((pairs) => {
      if (cancelled) return;
      const yearByPaper = new Map(pairs.map(([id, date]) => [id, date ? new Date(date).getFullYear() : null]));
      const counts: Record<string, number> = {};
      for (const item of evidence) {
        const year = yearByPaper.get(item.paper_id);
        if (year == null) continue;
        counts[year] = (counts[year] ?? 0) + 1;
      }
      setByYear(counts);
    });

    return () => {
      cancelled = true;
    };
  }, [evidence]);

  if (byYear === null) {
    return <p className="text-[0.875rem] text-[var(--ink-faint)]">Loading…</p>;
  }

  if (Object.keys(byYear).length === 0) {
    return <Unassessed reason="No supporting passage traces back to a paper with a known publication year." />;
  }

  return (
    <div>
      <IdeaYearLineChart byYear={byYear} />
      <p className="mt-3 max-w-[58ch] text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
        Supporting passages grouped by the publication year of the paper they came from - a proxy for when the
        retrieved literature was mostly discussing this idea, not a count of all papers ever written on it.
      </p>
    </div>
  );
}

const CHART_WIDTH = 640;
const CHART_HEIGHT = 160;
const CHART_PAD_X = 16;
const CHART_PAD_TOP = 14;
const CHART_PAD_BOTTOM = 28;

/** A thin monochrome line chart: count of supporting passages per year,
    zero-filled across the full range so the line reads as a continuous trend. */
function IdeaYearLineChart({ byYear }: { byYear: Record<string, number> }) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const observedYears = Object.keys(byYear)
    .map(Number)
    .sort((a, b) => a - b);
  const minYear = observedYears[0];
  const maxYear = observedYears[observedYears.length - 1];
  const years: number[] = [];
  for (let y = minYear; y <= maxYear; y++) years.push(y);

  const counts = years.map((y) => byYear[String(y)] ?? 0);
  const peak = Math.max(...counts, 1);

  const plotWidth = CHART_WIDTH - CHART_PAD_X * 2;
  const plotHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;

  const xFor = (i: number) => CHART_PAD_X + (years.length === 1 ? plotWidth / 2 : (i / (years.length - 1)) * plotWidth);
  const yFor = (count: number) => CHART_PAD_TOP + plotHeight - (count / peak) * plotHeight;

  const linePath = counts.map((count, i) => `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(count)}`).join(" ");
  const baselineY = CHART_PAD_TOP + plotHeight;

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        className="w-full"
        role="img"
        aria-label={`Supporting passages by year, ${minYear} to ${maxYear}`}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {/* recessive baseline */}
        <line
          x1={CHART_PAD_X}
          y1={baselineY}
          x2={CHART_WIDTH - CHART_PAD_X}
          y2={baselineY}
          stroke="var(--rule-soft)"
          strokeWidth={1}
        />

        <path d={linePath} fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />

        {years.map((year, i) => (
          <g key={year}>
            {/* generous invisible hit target, per interaction spec */}
            <rect
              x={xFor(i) - plotWidth / years.length / 2}
              y={0}
              width={plotWidth / years.length}
              height={CHART_HEIGHT}
              fill="transparent"
              onMouseEnter={() => setHoverIndex(i)}
            />
            <circle
              cx={xFor(i)}
              cy={yFor(counts[i])}
              r={hoverIndex === i ? 5 : 3}
              fill={counts[i] > 0 ? "var(--ink)" : "var(--rule)"}
            />
            <text
              x={xFor(i)}
              y={CHART_HEIGHT - 8}
              textAnchor="middle"
              className="readout"
              style={{ fontSize: "9px", fill: hoverIndex === i ? "var(--near)" : "var(--ink-faint)" }}
            >
              {String(year).slice(2)}
            </text>
          </g>
        ))}

        {hoverIndex !== null && (
          <line
            x1={xFor(hoverIndex)}
            y1={CHART_PAD_TOP}
            x2={xFor(hoverIndex)}
            y2={baselineY}
            stroke="var(--rule)"
            strokeWidth={1}
            strokeDasharray="2 2"
          />
        )}
      </svg>

      {hoverIndex !== null && (
        <div
          className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded-[2px] border border-[var(--rule)] bg-[var(--panel)] px-2 py-1 shadow-sm"
          style={{
            left: `${(xFor(hoverIndex) / CHART_WIDTH) * 100}%`,
            top: `${(yFor(counts[hoverIndex]) / CHART_HEIGHT) * 100}%`,
          }}
        >
          <p className="readout text-[0.6875rem] text-[var(--ink)] tabular-nums">
            {years[hoverIndex]} · {counts[hoverIndex]} passage{counts[hoverIndex] === 1 ? "" : "s"}
          </p>
        </div>
      )}
    </div>
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
  comparison_summary arrives as blocks the assessment builder assembled,
  one per question rather than one per paper (assessment/existing_solutions.py):

      Problems already addressed
      - "Paper Title": <claim text>
      - "Other Paper": <claim text>

      Existing approaches / methods used
      - "Paper Title": <claim text>

  research_gap and applications claims are deliberately never in here -
  they have their own dedicated report sections below, sourced from the
  same underlying evidence but with their own relevance gating.
*/
function ComparisonSummary({ text }: { text: string }) {
  const blocks = text.split("\n\n").filter(Boolean);

  return (
    <div className="max-w-[64ch] space-y-6">
      {blocks.map((block, blockIndex) => {
        const [heading, ...claimLines] = block.split("\n");
        return (
          <div key={blockIndex}>
            <p className="font-[family-name:var(--type-text)] text-[0.9375rem] font-semibold leading-snug">
              {heading}
            </p>
            <ul className="mt-2 space-y-2">
              {claimLines.map((line, lineIndex) => {
                const match = /^-\s*"([^"]*)":\s*(.*)$/.exec(line);
                if (!match) return null;
                const [, paperTitle, claimText] = match;
                return (
                  <li key={lineIndex}>
                    <p className="font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
                      {claimText}
                    </p>
                    <span className="eyebrow mt-0.5 inline-block">{paperTitle}</span>
                  </li>
                );
              })}
            </ul>
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

const TIER_LABEL: Record<PotentialOpportunity["tier"], string> = {
  direct: "direct",
  adjacent: "adjacent",
  speculative: "speculative",
};

/*
  The one field in this report synthesized by a local LLM rather than
  extracted - see docs/superpowers/specs/2026-09-03-opportunities-synthesis-
  design.md. Off unless the backend has OLLAMA_ENABLED=true (the button
  simply 503s, shown as an inline error, if not) - never generated during
  the initial assessment, only on explicit request here, and every citation
  the model produced was already checked against real application indices
  server-side before this ever renders.
*/
function Opportunities({
  assessmentId,
  applications,
  opportunities: initialOpportunities,
  onSynthesized,
}: {
  assessmentId: string;
  applications: PotentialApplication[] | null;
  opportunities: PotentialOpportunity[] | null;
  onSynthesized?: (updated: ResearchAssessment) => void;
}) {
  const [opportunities, setOpportunities] = useState(initialOpportunities);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  async function synthesize() {
    setBusy(true);
    setFailed(false);
    try {
      const updated = await assessmentApi.synthesizeOpportunities(assessmentId);
      setOpportunities(updated.potential_opportunities);
      onSynthesized?.(updated);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  if (!applications || applications.length === 0) {
    return (
      <Unassessed reason="Not enough evidence: no potential applications were found for this idea to synthesize opportunities from." />
    );
  }

  if (!opportunities) {
    return (
      <div>
        <Unassessed reason="Not generated yet. This is the one field synthesized by a local LLM, grounded only in the applications above - never invented from nothing." />
        <button
          type="button"
          onClick={synthesize}
          disabled={busy}
          className="eyebrow mt-3 rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-50"
        >
          {busy ? "synthesizing…" : "✨ synthesize opportunities"}
        </button>
        {failed && (
          <p className="mt-2 text-[0.75rem] text-[var(--live)]">
            local LLM unavailable — this section requires an operator to enable it
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      <span className="eyebrow text-[var(--ink-faint)]">
        AI-synthesized from the applications above — not independently verified
      </span>
      <ul className="mt-3 space-y-4">
        {opportunities.map((o) => (
          <li key={o.tier}>
            <span className="eyebrow">{TIER_LABEL[o.tier]}</span>
            <p className="mt-1 font-[family-name:var(--type-text)] text-[0.9375rem] leading-relaxed">
              {o.opportunity}
            </p>
            <p className="mt-1 text-[0.8125rem] text-[var(--ink-soft)]">
              from{" "}
              {o.source_applications.map((source, i) => (
                <span key={source.paper_id}>
                  {i > 0 && ", "}
                  <Link
                    href={`/papers/${source.paper_id}`}
                    className="underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
                  >
                    {source.paper_title}
                  </Link>
                </span>
              ))}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
