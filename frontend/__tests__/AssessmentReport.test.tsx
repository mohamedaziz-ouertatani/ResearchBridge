import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AssessmentReport } from "@/components/AssessmentReport";
import type { ResearchAssessment } from "@/lib/assessmentApi";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

function baseAssessment(overrides: Partial<ResearchAssessment> = {}): ResearchAssessment {
  return {
    id: "a1",
    research_input: {
      id: "ri1",
      input_type: "idea",
      raw_text: "A research idea about federated fraud detection.",
      title: null,
      matched_paper_id: null,
    },
    status: "completed",
    retrieved_paper_ids: [],
    comparison_summary: null,
    novelty_level: "high",
    novelty_reasoning: null,
    research_gap_text: null,
    research_gap_source: null,
    candidate_gap_id: null,
    potential_applications: null,
    technical_feasibility_level: "not_assessed",
    technical_feasibility_reasoning: null,
    potential_opportunities: null,
    risks_and_limitations: null,
    external_validation_needed: "Independent replication is needed before deployment.",
    recommendation: "worth pursuing",
    confidence: "medium",
    human_reviewed: false,
    evidence: [],
    ...overrides,
  };
}

// AssessmentHistory (a child of AssessmentReport) always fetches on mount -
// stub it out globally so every test resolves to "no sibling re-runs" and
// renders nothing from that section, which is what most cases below expect.
function stubHistoryFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve([]),
    } as Response),
  );
}

describe("AssessmentReport", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the recommendation and confidence from the assessment", async () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    expect(screen.getByText("worth pursuing")).toBeInTheDocument();
    expect(screen.getByText("medium")).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });

  it("shows an unassessed placeholder for every ungrounded field rather than leaving it blank", async () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    // comparison_summary, research_gap_text, potential_applications and
    // risks_and_limitations are all null in the fixture above - the report
    // must never render an empty section for a null field
    expect(screen.getByText(/No retrieved paper had extracted claims to compare against/)).toBeInTheDocument();
    expect(screen.getByText(/No gap was found in the retrieved literature/)).toBeInTheDocument();
    expect(screen.getByText(/No retrieved paper stated a limitation/)).toBeInTheDocument();
  });

  it("renders a matched-paper link only when the input has a matched_paper_id", async () => {
    stubHistoryFetch();
    const { rerender } = render(<AssessmentReport assessment={baseAssessment()} />);
    expect(screen.queryByText("view it")).not.toBeInTheDocument();

    rerender(
      <AssessmentReport
        assessment={baseAssessment({
          research_input: {
            id: "ri1",
            input_type: "document",
            raw_text: "text",
            title: null,
            matched_paper_id: "paper-123",
          },
        })}
      />,
    );
    expect(screen.getByText("view it")).toBeInTheDocument();
  });

  it("lists potential applications with a link back to their source paper", async () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          potential_applications: [
            { application: "real-time fraud screening", source_paper: "Fraud Paper", paper_id: "p1" },
          ],
        })}
      />,
    );

    expect(screen.getByText("real-time fraud screening")).toBeInTheDocument();
    expect(screen.getByText("Fraud Paper")).toBeInTheDocument();
  });
});
