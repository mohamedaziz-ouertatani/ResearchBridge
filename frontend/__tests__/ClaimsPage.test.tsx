import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import Claims from "@/app/claims/page";
import {
  claimsApi,
  extractedClaimsApi,
  type AnalysisClaimPage,
  type ExtractedClaimPage,
} from "@/lib/claimsApi";

vi.mock("next/navigation", () => ({
  usePathname: () => "/claims",
}));

const emptyAnalysisPage: AnalysisClaimPage = {
  items: [],
  total: 0,
  limit: 20,
  offset: 0,
};

const extractedPage: ExtractedClaimPage = {
  items: [
    {
      id: "e1",
      claim_type: "applications",
      text: "Deployed via a WeChat Mini Program for loan officers.",
      confidence: "medium",
      section: "discussion",
      extraction_method: "hybrid",
      paper_id: "p1",
      paper_title: "A Paper About Fraud Detection",
      created_at: "2026-09-10T00:00:00Z",
    },
  ],
  total: 1,
  limit: 20,
  offset: 0,
};

describe("Claims page", () => {
  afterEach(() => vi.restoreAllMocks());

  it("defaults to the reasoning-claims view", async () => {
    vi.spyOn(claimsApi, "list").mockResolvedValue(emptyAnalysisPage);

    render(<Claims />);

    await waitFor(() =>
      expect(screen.getByText("No claims match these filters.")).toBeInTheDocument(),
    );
    expect(claimsApi.list).toHaveBeenCalled();
  });

  it("switches to the extraction-claims view and shows a type filter", async () => {
    vi.spyOn(claimsApi, "list").mockResolvedValue(emptyAnalysisPage);
    vi.spyOn(extractedClaimsApi, "list").mockResolvedValue(extractedPage);

    render(<Claims />);
    await waitFor(() => expect(claimsApi.list).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: /extraction claims/i }));

    await waitFor(() =>
      expect(screen.getByText("Deployed via a WeChat Mini Program for loan officers.")).toBeInTheDocument(),
    );
    expect(screen.getByText("A Paper About Fraud Detection")).toBeInTheDocument();
    expect(screen.getByText("applications · confidence: medium")).toBeInTheDocument();
    expect(extractedClaimsApi.list).toHaveBeenCalledWith({ claim_type: "all" }, 20, 0);
  });

  it("re-queries extracted claims when the type filter changes", async () => {
    vi.spyOn(claimsApi, "list").mockResolvedValue(emptyAnalysisPage);
    vi.spyOn(extractedClaimsApi, "list").mockResolvedValue(extractedPage);

    render(<Claims />);
    fireEvent.click(screen.getByRole("button", { name: /extraction claims/i }));
    await waitFor(() => expect(extractedClaimsApi.list).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText("type"), { target: { value: "applications" } });

    await waitFor(() =>
      expect(extractedClaimsApi.list).toHaveBeenLastCalledWith(
        { claim_type: "applications" },
        20,
        0,
      ),
    );
  });
});
