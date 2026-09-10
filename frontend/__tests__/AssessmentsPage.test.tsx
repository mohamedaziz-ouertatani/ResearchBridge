import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import AssessmentDashboard from "@/app/assessments/page";
import { assessmentApi, type AssessmentSummary, type AssessmentSummaryPage } from "@/lib/assessmentApi";

vi.mock("next/navigation", () => ({
  usePathname: () => "/assessments",
}));

function summary(id: string, overrides: Partial<AssessmentSummary> = {}): AssessmentSummary {
  return {
    id,
    created_at: "2026-09-10T00:00:00Z",
    status: "completed",
    novelty_level: "low",
    technical_feasibility_level: "medium",
    recommendation: "LOW PRIORITY",
    confidence: "medium",
    human_reviewed: false,
    research_input_id: `ri-${id}`,
    input_type: "idea",
    input_preview: `Idea ${id}`,
    ...overrides,
  };
}

function page(items: AssessmentSummary[], total: number, offset = 0): AssessmentSummaryPage {
  return { items, total, limit: 50, offset };
}

describe("Assessments page pagination", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows a count and no pagination controls when everything fits on one page", async () => {
    vi.spyOn(assessmentApi, "list").mockResolvedValue(page([summary("a1"), summary("a2")], 2));

    render(<AssessmentDashboard />);

    await waitFor(() => expect(screen.getByText("Idea a1")).toBeInTheDocument());
    expect(screen.getByText("1–2 of 2")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "next" })).not.toBeInTheDocument();
  });

  it("shows next/previous controls when more results exist than fit on one page, and pages through them", async () => {
    const listSpy = vi
      .spyOn(assessmentApi, "list")
      .mockResolvedValueOnce(page([summary("a1")], 120, 0))
      .mockResolvedValueOnce(page([summary("a2")], 120, 50));

    render(<AssessmentDashboard />);

    await waitFor(() => expect(screen.getByText("1–50 of 120")).toBeInTheDocument());
    const nextButton = screen.getByRole("button", { name: "next" });
    expect(nextButton).not.toBeDisabled();

    fireEvent.click(nextButton);

    await waitFor(() => expect(screen.getByText("51–100 of 120")).toBeInTheDocument());
    expect(listSpy).toHaveBeenLastCalledWith(
      "needs_review",
      { sort: "newest", novelty: undefined, feasibility: undefined },
      50,
      50,
    );
  });

  it("resets to the first page when a filter changes", async () => {
    const listSpy = vi
      .spyOn(assessmentApi, "list")
      .mockResolvedValueOnce(page([summary("a1")], 120, 0))
      .mockResolvedValueOnce(page([summary("a2")], 120, 50))
      .mockResolvedValueOnce(page([summary("a3")], 5, 0));

    render(<AssessmentDashboard />);
    await waitFor(() => expect(screen.getByText("1–50 of 120")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "next" }));
    await waitFor(() => expect(screen.getByText("51–100 of 120")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "all" }));

    await waitFor(() =>
      expect(listSpy).toHaveBeenLastCalledWith(
        "all",
        { sort: "newest", novelty: undefined, feasibility: undefined },
        50,
        0,
      ),
    );
  });
});
