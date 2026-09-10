import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { ClaimNode, type ClaimFlowNode } from "@/components/graph/ClaimNode";

function props(data: ClaimFlowNode["data"]): NodeProps<ClaimFlowNode> {
  return { id: "claim-1", type: "claim", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<ClaimFlowNode>;
}

describe("ClaimNode", () => {
  it("renders the claim text, type, confidence, and status", () => {
    render(
      <ReactFlowProvider>
        <ClaimNode {...props({ label: "The method generalizes.", claim_type: "inference", confidence: "medium", status: "pending" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("The method generalizes.")).toBeInTheDocument();
    expect(screen.getByText("claim · inference")).toBeInTheDocument();
    expect(screen.getByText(/medium confidence/)).toBeInTheDocument();
    expect(screen.getByText(/pending/)).toBeInTheDocument();
  });

  it("omits the type suffix when claim_type is null", () => {
    render(
      <ReactFlowProvider>
        <ClaimNode {...props({ label: "A claim.", claim_type: null, confidence: null, status: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("claim")).toBeInTheDocument();
  });
});
