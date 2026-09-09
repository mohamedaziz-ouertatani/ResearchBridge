import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AssessmentComparison } from "@/components/AssessmentComparison";
import type { ResearchAssessment } from "@/lib/assessmentApi";

vi.mock("next/navigation", () => ({
  usePathname: () => "/assessments/compare",
}));

function assessment(
  id: string,
  overrides: Partial<ResearchAssessment> = {},
): ResearchAssessment {
  return {
    id,
    created_at: "2026-09-06T10:00:00Z",
    research_input: {
      id: `input-${id}`,
      input_type: "idea",
      raw_text: `Research idea ${id}`,
      title: null,
      matched_paper_id: null,
    },
    status: "completed",
    retrieved_paper_ids: ["shared-paper", `paper-${id}`],
    comparison_summary: null,
    novelty_level: "high",
    novelty_reasoning: null,
    research_gap_text: `Gap for ${id}`,
    research_gap_source: "input_specific",
    candidate_gap_id: null,
    potential_applications: null,
    potential_applications_status: "not_assessed",
    technical_feasibility_level: "medium",
    technical_feasibility_reasoning: null,
    potential_opportunities: null,
    risks_and_limitations: null,
    recommendation: id === "a1" ? "worth pursuing" : "requires human review",
    confidence: "medium",
    human_reviewed: id === "a1",
    evidence: [
      {
        role: "novelty",
        paper_id: "shared-paper",
        paper_title: "Shared paper",
        text: `Evidence for ${id}`,
        section: "results",
      },
    ],
    claims: [],
    ...overrides,
  };
}

describe("AssessmentComparison", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads selected reports and compares their evidence and outcomes", async () => {
    const reports = { a1: assessment("a1"), a2: assessment("a2") };
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        const id = url.includes("/a1") ? "a1" : "a2";
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(reports[id as keyof typeof reports]),
        } as Response);
      }),
    );

    render(<AssessmentComparison ids={["a1", "a2"]} />);

    await waitFor(() =>
      expect(screen.getByText("shared retrieved papers")).toBeInTheDocument(),
    );
    expect(screen.getByText("shared evidence papers")).toBeInTheDocument();
    expect(screen.getAllByText("1").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("worth pursuing")).toBeInTheDocument();
    expect(screen.getByText("requires human review")).toBeInTheDocument();
    expect(screen.getByText("Gap for a1")).toBeInTheDocument();
    expect(screen.getByText("Gap for a2")).toBeInTheDocument();
  });

  it("asks the user to select at least two assessments", () => {
    render(<AssessmentComparison ids={["a1"]} />);

    expect(
      screen.getByText(/Choose at least two completed assessments/),
    ).toBeInTheDocument();
  });
});
