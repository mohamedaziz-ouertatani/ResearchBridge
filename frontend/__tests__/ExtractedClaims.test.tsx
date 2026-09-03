import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ExtractedClaims } from "@/components/ExtractedClaims";
import type { ExtractedClaim } from "@/lib/api";

function claim(overrides: Partial<ExtractedClaim> = {}): ExtractedClaim {
  return {
    claim_type: "problem",
    text: "The paper addresses X.",
    confidence: "low",
    section: null,
    extraction_method: "heuristic",
    ...overrides,
  };
}

describe("ExtractedClaims", () => {
  it("renders nothing when there are no claims", () => {
    const { container } = render(<ExtractedClaims claims={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("shows a human-readable label for a known claim type", () => {
    render(<ExtractedClaims claims={[claim({ claim_type: "research_gap" })]} />);

    expect(screen.getByText("Research gap")).toBeInTheDocument();
  });

  it("falls back to the raw claim_type string for an unknown type", () => {
    render(<ExtractedClaims claims={[claim({ claim_type: "some_new_field" })]} />);

    expect(screen.getByText("some_new_field")).toBeInTheDocument();
  });

  it("shows the claim's text and its self-reported confidence, not a derived trust score", () => {
    render(<ExtractedClaims claims={[claim({ text: "A specific claim.", confidence: "medium" })]} />);

    expect(screen.getByText("A specific claim.")).toBeInTheDocument();
    expect(screen.getByText("medium confidence")).toBeInTheDocument();
  });

  it("renders one item per claim", () => {
    render(
      <ExtractedClaims
        claims={[claim({ claim_type: "problem" }), claim({ claim_type: "method", text: "Uses X." })]}
      />,
    );

    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });
});
