import { afterEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ThemeToggle } from "@/components/ThemeToggle";

afterEach(() => {
  document.documentElement.removeAttribute("data-theme");
  localStorage.clear();
});

describe("ThemeToggle", () => {
  it("reads the theme the inline script already applied via data-theme, defaulting to light", () => {
    render(<ThemeToggle />);

    // light theme -> offers to switch to dark
    expect(screen.getByRole("button", { name: "Switch to dark theme" })).toHaveTextContent("dark");
  });

  it("picks up an existing dark data-theme attribute on mount", () => {
    document.documentElement.setAttribute("data-theme", "dark");

    render(<ThemeToggle />);

    expect(screen.getByRole("button", { name: "Switch to light theme" })).toHaveTextContent("light");
  });

  it("clicking toggles the data-theme attribute and the button's own label", () => {
    render(<ThemeToggle />);

    fireEvent.click(screen.getByRole("button"));

    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(screen.getByRole("button", { name: "Switch to light theme" })).toHaveTextContent("light");
  });

  it("clicking twice returns to light", () => {
    render(<ThemeToggle />);
    const button = screen.getByRole("button");

    fireEvent.click(button);
    fireEvent.click(button);

    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("persists the chosen theme to localStorage", () => {
    render(<ThemeToggle />);

    fireEvent.click(screen.getByRole("button"));

    expect(localStorage.getItem("rb-theme")).toBe("dark");
  });

  it("still toggles the on-screen theme even if localStorage throws (private mode/quota)", () => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = () => {
      throw new Error("quota exceeded");
    };

    try {
      render(<ThemeToggle />);
      fireEvent.click(screen.getByRole("button"));

      expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    } finally {
      Storage.prototype.setItem = original;
    }
  });
});
