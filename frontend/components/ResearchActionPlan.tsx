import Link from "next/link";
import type {
  AssessmentEvidence,
  ResearchAssessment,
} from "@/lib/assessmentApi";

type ActionPlanStep = {
  id: string;
  title: string;
  objective: string;
  actions: string[];
  evidence: AssessmentEvidence[];
};

const ROLE_LABELS: Record<string, string> = {
  comparison: "existing-solution evidence",
  research_gap: "research-gap evidence",
  feasibility: "feasibility evidence",
  application: "application evidence",
  risk: "risk evidence",
};

export function buildResearchActionPlan(
  assessment: ResearchAssessment,
): ActionPlanStep[] {
  const evidenceByRole = new Map<string, AssessmentEvidence[]>();
  for (const item of assessment.evidence) {
    const items = evidenceByRole.get(item.role) ?? [];
    items.push(item);
    evidenceByRole.set(item.role, items);
  }

  const evidenceFor = (
    role: AssessmentEvidence["role"],
  ): AssessmentEvidence[] => evidenceByRole.get(role) ?? [];

  const comparisonEvidence = evidenceFor("comparison");
  const gapEvidence = evidenceFor("research_gap");
  const feasibilityEvidence = evidenceFor("feasibility");
  const applicationEvidence = evidenceFor("application");
  const riskEvidence = evidenceFor("risk");

  return [
    {
      id: "baseline",
      title: "Reproduce the closest baseline",
      objective:
        "Establish whether the proposed idea improves on the approaches already found in the corpus.",
      actions: [
        "Choose the closest existing method from the comparison evidence.",
        "Recreate its evaluation setup before changing the method.",
        "Record the baseline result and the exact change your idea introduces.",
      ],
      evidence: comparisonEvidence,
    },
    {
      id: "gap",
      title: "Turn the research gap into a testable claim",
      objective: assessment.research_gap_text
        ? `Test whether the reported gap is real: ${assessment.research_gap_text}`
        : "A grounded gap has not been established for this assessment.",
      actions: [
        "Write one falsifiable statement describing what existing work cannot yet do.",
        "Define the observation that would count as addressing that statement.",
        "Stop and refine the question if the available evidence does not support a specific gap.",
      ],
      evidence: assessment.research_gap_text ? gapEvidence : [],
    },
    {
      id: "prototype",
      title: "Build the smallest feasibility prototype",
      objective: assessment.technical_feasibility_reasoning
        ? `Use the documented technical grounding as the starting constraint: ${assessment.technical_feasibility_reasoning}`
        : "A grounded feasibility signal is not available yet.",
      actions: [
        "Implement only the smallest path needed to exercise the proposed contribution.",
        "Use the datasets, methods, or constraints described in the supporting papers.",
        "Measure runtime, data requirements, and failure cases before expanding scope.",
      ],
      evidence: feasibilityEvidence,
    },
    {
      id: "application",
      title: "Validate one target application",
      objective: assessment.potential_applications?.length
        ? `Start with the most concrete stated application: ${assessment.potential_applications[0].application}`
        : "No retrieved paper supplied a grounded application to validate.",
      actions: [
        "Choose one user, workflow, or operational setting represented by the evidence.",
        "Define a success measure that matters in that setting, not only a research metric.",
        "Test the prototype with a small representative case before making product claims.",
      ],
      evidence: applicationEvidence,
    },
    {
      id: "risks",
      title: "Stress-test the known limitations",
      objective: assessment.risks_and_limitations
        ? "The retrieved literature reports limitations that should become explicit validation cases."
        : "No grounded limitations were extracted for this assessment.",
      actions: [
        "Turn each reported limitation into a failure case or boundary condition.",
        "Document which risks are inherited from prior work and which are introduced by the new idea.",
        "Revisit the recommendation only after the highest-impact risks have an observed result.",
      ],
      evidence: riskEvidence,
    },
  ];
}

export function ResearchActionPlan({
  assessment,
}: {
  assessment: ResearchAssessment;
}) {
  const steps = buildResearchActionPlan(assessment);

  return (
    <div className="max-w-[68ch]">
      <p className="text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
        A practical sequence derived from this assessment. It does not invent a
        new research claim: each ready step is anchored to the passages shown
        below it.
      </p>

      <ol className="mt-5 space-y-5">
        {steps.map((step, index) => {
          const ready = step.evidence.length > 0;
          return (
            <li key={step.id} className="border-l border-[var(--rule)] pl-4">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <h3 className="font-[family-name:var(--type-text)] text-[0.9375rem] font-semibold">
                  <span className="readout mr-2 text-[0.75rem] text-[var(--ink-faint)]">
                    0{index + 1}
                  </span>
                  {step.title}
                </h3>
                <span className="eyebrow text-[var(--ink-faint)]">
                  {ready
                    ? `${step.evidence.length} supporting passage${step.evidence.length === 1 ? "" : "s"}`
                    : "blocked"}
                </span>
              </div>

              <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
                {step.objective}
              </p>

              {ready ? (
                <ul className="mt-3 list-decimal space-y-1 pl-5 text-[0.8125rem] leading-relaxed text-[var(--ink)]">
                  {step.actions.map((action) => (
                    <li key={action} className="pl-1">
                      {action}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-3 text-[0.8125rem] leading-relaxed text-[var(--ink-faint)]">
                  Blocked until the corpus contains a supporting passage for
                  this checkpoint. Do not treat this step as a validated
                  recommendation yet.
                </p>
              )}

              {ready && (
                <details className="mt-3">
                  <summary className="eyebrow cursor-pointer hover:text-[var(--ink)]">
                    show grounding ·{" "}
                    {ROLE_LABELS[step.evidence[0].role] ??
                      "assessment evidence"}
                  </summary>
                  <ul className="mt-3 space-y-3 border-l border-[var(--rule-soft)] pl-4">
                    {step.evidence.map((item, evidenceIndex) => (
                      <li key={`${item.paper_id}-${evidenceIndex}`}>
                        <p className="font-[family-name:var(--type-text)] text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
                          “{item.text}”
                        </p>
                        <Link
                          href={`/papers/${item.paper_id}`}
                          className="eyebrow mt-1 inline-block hover:text-[var(--ink)]"
                        >
                          {item.paper_title}
                          {item.section ? ` · ${item.section}` : ""}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
