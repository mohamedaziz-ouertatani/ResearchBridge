import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { GaugeLegend, ProximityGauge } from "@/components/ProximityGauge";

describe("ProximityGauge", () => {
  it("prints the distance to three decimal places", () => {
    render(<ProximityGauge distance={0.531} />);

    expect(screen.getByText("0.531")).toBeInTheDocument();
  });

  it("labels the gauge's accessible role with the raw distance and scale", () => {
    render(<ProximityGauge distance={0.25} />);

    expect(
      screen.getByRole("img", { name: "Semantic distance 0.250 of a possible 1.0; lower is closer in meaning" }),
    ).toBeInTheDocument();
  });

  it("clamps a distance above the scale max to 100% needle position", () => {
    render(<ProximityGauge distance={1.4} />);

    // the needle is the only positioned element with an inline "background"
    // (its color) alongside "left" - ticks/end-caps only set "left"
    const needle = screen.getByRole("img").querySelector('[style*="background"]') as HTMLElement;
    expect(needle.style.left).toBe("100%");
  });

  it("clamps a negative distance (real cosine-distance data can dip slightly below 0) to 0% needle position", () => {
    render(<ProximityGauge distance={-0.077} />);

    const needle = screen.getByRole("img").querySelector('[style*="background"]') as HTMLElement;
    expect(needle.style.left).toBe("0%");
  });

  it("still prints the raw (unclamped) distance value even when the needle position is clamped", () => {
    render(<ProximityGauge distance={1.4} />);

    expect(screen.getByText("1.400")).toBeInTheDocument();
  });
});

describe("GaugeLegend", () => {
  it("renders the closer/further labels", () => {
    render(<GaugeLegend />);

    expect(screen.getByText("closer")).toBeInTheDocument();
    expect(screen.getByText("further")).toBeInTheDocument();
  });
});
