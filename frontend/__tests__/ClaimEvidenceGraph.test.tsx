import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ClaimEvidenceGraph } from "@/components/graph/ClaimEvidenceGraph";
import { assessmentApi } from "@/lib/assessmentApi";

describe("ClaimEvidenceGraph", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches the claim graph and renders each node's label", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "claim-1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
          confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
          gap_status: null, categories: [],
        },
        {
          id: "evidence-1", kind: "evidence", label: "a quoted passage", claim_type: null, confidence: null,
          status: null, paper_id: "p1", paper_title: "Some Paper", section: "results", gap_status: null,
          categories: [],
        },
      ],
      edges: [{ source: "claim-1", target: "evidence-1", relationship: "supports" }],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getByText("The method generalizes.")).toBeInTheDocument());
    expect(screen.getByText('"a quoted passage"')).toBeInTheDocument();
  });

  it("hides gap nodes when the gaps filter is unchecked", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "gap-1", kind: "gap", label: "a recurring unaddressed limitation", claim_type: null,
          confidence: null, status: null, paper_id: null, paper_title: null, section: null,
          gap_status: "strong_gap", categories: [],
        },
      ],
      edges: [],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getByText("a recurring unaddressed limitation")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("checkbox", { name: "gaps" }));

    expect(screen.queryByText("a recurring unaddressed limitation")).not.toBeInTheDocument();
  });

  it("opens the inspector drawer with the clicked node's detail", async () => {
    vi.spyOn(assessmentApi, "claimGraph").mockResolvedValue({
      nodes: [
        {
          id: "claim-1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
          confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
          gap_status: null, categories: [],
        },
      ],
      edges: [],
    });
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({ buckets: [] });

    render(<ClaimEvidenceGraph assessmentId="a1" />);

    await waitFor(() => expect(screen.getAllByText("The method generalizes.")).toHaveLength(1));

    fireEvent.click(screen.getByText("The method generalizes."));

    await waitFor(() => expect(screen.getAllByText("The method generalizes.")).toHaveLength(2));
    expect(screen.getByText("confidence: medium")).toBeInTheDocument();
  });
});
