import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  ResearchActionPlan,
  buildResearchActionPlan,
} from "@/components/ResearchActionPlan";
import type { ResearchAssessment } from "@/lib/assessmentApi";

function assessment(
  overrides: Partial<ResearchAssessment> = {},
): ResearchAssessment {
  return {
    id: "assessment-1",
    research_input: {
      id: "input-1",
      input_type: "idea",
      raw_text: "A research idea",
      title: null,
      matched_paper_id: null,
    },
    status: "completed",
    retrieved_paper_ids: ["paper-1"],
    comparison_summary: "An existing approach",
    novelty_level: "medium",
    novelty_reasoning: null,
    research_gap_text: "Existing work does not test this setting.",
    research_gap_source: "input_specific",
    candidate_gap_id: null,
    potential_applications: [
      {
        application: "fraud screening",
        source_paper: "Paper One",
        paper_id: "paper-1",
      },
    ],
    potential_applications_status: "found",
    technical_feasibility_level: "medium",
    technical_feasibility_reasoning:
      "A related method is already implemented in the literature.",
    potential_opportunities: null,
    risks_and_limitations: "The approach is limited by sparse labels.",
    recommendation: "worth pursuing",
    confidence: "medium",
    human_reviewed: false,
    evidence: [
      {
        role: "comparison",
        paper_id: "paper-1",
        paper_title: "Paper One",
        text: "Existing baseline.",
        section: "methods",
      },
      {
        role: "research_gap",
        paper_id: "paper-1",
        paper_title: "Paper One",
        text: "The gap remains.",
        section: "conclusion",
      },
      {
        role: "feasibility",
        paper_id: "paper-1",
        paper_title: "Paper One",
        text: "A method exists.",
        section: "methods",
      },
      {
        role: "application",
        paper_id: "paper-1",
        paper_title: "Paper One",
        text: "Useful for screening.",
        section: "discussion",
      },
      {
        role: "risk",
        paper_id: "paper-1",
        paper_title: "Paper One",
        text: "Labels are sparse.",
        section: "limitations",
      },
    ],
    claims: [],
    ...overrides,
  };
}

describe("ResearchActionPlan", () => {
  it("creates ready, evidence-backed checkpoints", () => {
    const steps = buildResearchActionPlan(assessment());

    expect(steps).toHaveLength(5);
    expect(steps.every((step) => step.evidence.length > 0)).toBe(true);
    expect(steps[0].actions).toContain(
      "Choose the closest existing method from the comparison evidence.",
    );
  });

  it("blocks checkpoints when the assessment has no grounding", () => {
    const steps = buildResearchActionPlan(
      assessment({
        research_gap_text: null,
        potential_applications: null,
        technical_feasibility_reasoning: null,
        risks_and_limitations: null,
        evidence: [],
      }),
    );

    expect(steps.every((step) => step.evidence.length === 0)).toBe(true);
  });

  it("renders supporting passages under the action plan", () => {
    render(<ResearchActionPlan assessment={assessment()} />);

    expect(
      screen.getByText("Reproduce the closest baseline"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/A practical sequence derived from this assessment/),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/show grounding/)).toHaveLength(5);
  });
});
