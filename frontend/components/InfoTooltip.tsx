"use client";

import { useState } from "react";

/*
  A small "what does this do" glyph for a feature/control, not a full help
  system - one short explanation, revealed on hover or keyboard focus so it
  works without a mouse. Kept separate from native `title` because `title`
  can't be styled to match the faceplate and has an inconsistent delay
  across browsers.
*/
export function InfoTooltip({ text, label }: { text: string; label?: string }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label={label ?? "What does this do?"}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
        className="flex h-[1.05em] w-[1.05em] items-center justify-center rounded-full border border-[var(--ink-faint)] font-[family-name:var(--type-mono)] text-[0.625rem] leading-none text-[var(--ink-faint)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
      >
        i
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-[calc(100%+0.4rem)] left-1/2 z-10 w-max max-w-[18rem] -translate-x-1/2 rounded-[2px] border border-[var(--rule)] bg-[var(--panel)] px-3 py-2 text-[0.75rem] font-normal leading-relaxed text-[var(--ink-soft)] shadow-sm"
        >
          {text}
        </span>
      )}
    </span>
  );
}
