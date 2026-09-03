import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { InfoTooltip } from "@/components/InfoTooltip";

describe("InfoTooltip", () => {
  it("does not show the tooltip text until hovered or focused", () => {
    render(<InfoTooltip text="Explains the thing." />);

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows the tooltip on mouse enter and hides it on mouse leave", () => {
    render(<InfoTooltip text="Explains the thing." />);
    const trigger = screen.getByRole("button");

    fireEvent.mouseEnter(trigger);
    expect(screen.getByRole("tooltip")).toHaveTextContent("Explains the thing.");

    fireEvent.mouseLeave(trigger);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows the tooltip on keyboard focus and hides it on blur", () => {
    render(<InfoTooltip text="Explains the thing." />);
    const trigger = screen.getByRole("button");

    fireEvent.focus(trigger);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    fireEvent.blur(trigger);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("uses a default accessible label when none is given", () => {
    render(<InfoTooltip text="x" />);

    expect(screen.getByRole("button", { name: "What does this do?" })).toBeInTheDocument();
  });

  it("uses a custom accessible label when given", () => {
    render(<InfoTooltip text="x" label="What is confidence?" />);

    expect(screen.getByRole("button", { name: "What is confidence?" })).toBeInTheDocument();
  });

  it("clicking the trigger does not navigate or bubble (preventDefault/stopPropagation)", () => {
    let bubbled = false;
    render(
      <div onClick={() => (bubbled = true)}>
        <InfoTooltip text="x" />
      </div>,
    );

    fireEvent.click(screen.getByRole("button"));

    expect(bubbled).toBe(false);
  });
});
