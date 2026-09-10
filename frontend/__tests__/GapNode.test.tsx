import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { GapNode, type GapFlowNode } from "@/components/graph/GapNode";

function props(data: GapFlowNode["data"]): NodeProps<GapFlowNode> {
  return { id: "gap-1", type: "gap", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<GapFlowNode>;
}

describe("GapNode", () => {
  it("renders the observation and a human-readable gap status label", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "a recurring unaddressed limitation", gap_status: "strong_gap" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("a recurring unaddressed limitation")).toBeInTheDocument();
    expect(screen.getByText("strong gap")).toBeInTheDocument();
  });

  it("renders without a status badge when gap_status is null", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "an observation", gap_status: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("an observation")).toBeInTheDocument();
    expect(screen.queryByText("strong gap")).not.toBeInTheDocument();
  });

  it("applies stronger emphasis when highlighted by the density overlay", () => {
    render(
      <ReactFlowProvider>
        <GapNode {...props({ label: "an observation", gap_status: null, highlighted: true })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("an observation").closest("div")?.className).toContain("border-[var(--ink)]");
  });
});
