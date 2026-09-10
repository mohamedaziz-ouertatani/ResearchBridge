"use client";

import type { ClaimGraphNode } from "@/lib/assessmentApi";
import { GAP_STATUS_LABELS } from "@/lib/gapsApi";

export type InspectorDrawerProps = {
  node: ClaimGraphNode | null;
  onClose: () => void;
};

export function InspectorDrawer({ node, onClose }: InspectorDrawerProps) {
  if (!node) {
    return null;
  }

  return (
    <aside className="absolute right-0 top-0 h-full w-[340px] overflow-y-auto border-l border-[var(--rule)] bg-[var(--panel)] p-6">
      <button type="button" onClick={onClose} className="eyebrow text-[var(--ink-faint)] hover:text-[var(--ink)]">
        close
      </button>
      <span className="eyebrow mt-4 block text-[0.625rem] text-[var(--ink-faint)]">{node.kind}</span>
      <p className="mt-2 text-[0.9375rem] leading-relaxed text-[var(--ink)]">{node.label}</p>

      {node.kind === "claim" && (
        <dl className="mt-4 space-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
          {node.claim_type && <div>type: {node.claim_type}</div>}
          {node.confidence && <div>confidence: {node.confidence}</div>}
          {node.status && <div>status: {node.status}</div>}
        </dl>
      )}

      {node.kind === "evidence" && (
        <dl className="mt-4 space-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
          {node.paper_title && <div>paper: {node.paper_title}</div>}
          {node.section && <div>section: {node.section}</div>}
        </dl>
      )}

      {node.kind === "gap" && node.gap_status && (
        <span className="eyebrow mt-4 inline-block rounded-[2px] border border-[var(--rule)] px-2 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]">
          {GAP_STATUS_LABELS[node.gap_status]}
        </span>
      )}
    </aside>
  );
}
