import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InspectorDrawer } from "@/components/graph/InspectorDrawer";
import type { ClaimGraphNode } from "@/lib/assessmentApi";

function claimNode(overrides: Partial<ClaimGraphNode> = {}): ClaimGraphNode {
  return {
    id: "c1", kind: "claim", label: "The method generalizes.", claim_type: "inference",
    confidence: "medium", status: "pending", paper_id: null, paper_title: null, section: null,
    gap_status: null, categories: [], ...overrides,
  };
}

describe("InspectorDrawer", () => {
  it("renders nothing when no node is selected", () => {
    const { container } = render(<InspectorDrawer node={null} onClose={() => {}} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("shows claim details for a claim node", () => {
    render(<InspectorDrawer node={claimNode()} onClose={() => {}} />);

    expect(screen.getByText("The method generalizes.")).toBeInTheDocument();
    expect(screen.getByText("confidence: medium")).toBeInTheDocument();
    expect(screen.getByText("status: pending")).toBeInTheDocument();
  });

  it("shows evidence details for an evidence node", () => {
    render(
      <InspectorDrawer
        node={claimNode({ kind: "evidence", label: "a quoted passage", paper_title: "Some Paper", section: "results" })}
        onClose={() => {}}
      />,
    );

    expect(screen.getByText("paper: Some Paper")).toBeInTheDocument();
    expect(screen.getByText("section: results")).toBeInTheDocument();
  });

  it("shows a gap status badge for a gap node", () => {
    render(
      <InspectorDrawer node={claimNode({ kind: "gap", label: "an observation", gap_status: "strong_gap" })} onClose={() => {}} />,
    );

    expect(screen.getByText("strong gap")).toBeInTheDocument();
  });

  it("calls onClose when the close control is clicked", () => {
    const onClose = vi.fn();
    render(<InspectorDrawer node={claimNode()} onClose={onClose} />);

    fireEvent.click(screen.getByRole("button", { name: "close" }));

    expect(onClose).toHaveBeenCalledOnce();
  });
});
