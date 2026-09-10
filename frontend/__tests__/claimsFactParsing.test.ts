import { describe, expect, it } from "vitest";
import { parseFactBlocks } from "@/app/claims/page";

describe("parseFactBlocks", () => {
  it("parses a risks_and_limitations-style block (no heading, quoted text)", () => {
    const text =
      '- AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning: "On bf16 hardware the issue does not arise."\n' +
      '- MiCoPro: End-to-End Mixed Precision HW/SW Co-design with HW-aware Proxy Model: "However, existing algorithms for exploring MPQ schemes are limited in flexibility and efficiency."';

    expect(parseFactBlocks(text)).toEqual([
      {
        heading: null,
        items: [
          {
            title: "AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning",
            text: "On bf16 hardware the issue does not arise.",
          },
          {
            title: "MiCoPro: End-to-End Mixed Precision HW/SW Co-design with HW-aware Proxy Model",
            text: "However, existing algorithms for exploring MPQ schemes are limited in flexibility and efficiency.",
          },
        ],
      },
    ]);
  });

  it("keeps a wrapped quote's embedded newline inside one item instead of splitting it into two", () => {
    const text =
      '- Large Models for Small Devices: "However,\nthese techniques are rarely selected using a unified prefill-decode-level analysis."';

    expect(parseFactBlocks(text)).toEqual([
      {
        heading: null,
        items: [
          {
            title: "Large Models for Small Devices",
            text: "However,\nthese techniques are rarely selected using a unified prefill-decode-level analysis.",
          },
        ],
      },
    ]);
  });

  it("parses a comparison_summary-style block (heading, quoted title, unquoted text)", () => {
    const text =
      "Problems already addressed\n" +
      '- "AQLoRA": Quantized fine-tuning (QLoRA) saves memory but not time.\n\n' +
      "Existing approaches / methods used\n" +
      '- "MiCoPro": Quantized Neural Networks have low bitwidth.';

    expect(parseFactBlocks(text)).toEqual([
      {
        heading: "Problems already addressed",
        items: [{ title: "AQLoRA", text: "Quantized fine-tuning (QLoRA) saves memory but not time." }],
      },
      {
        heading: "Existing approaches / methods used",
        items: [{ title: "MiCoPro", text: "Quantized Neural Networks have low bitwidth." }],
      },
    ]);
  });

  it("returns null for plain prose that isn't a bulleted list, so callers fall back to plain rendering", () => {
    expect(parseFactBlocks("Only 0 of 7 dimensions have well-established evidence.")).toBeNull();
  });
});
