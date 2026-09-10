"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";
import { GAP_STATUS_LABELS } from "@/lib/gapsApi";

export type GapNodeData = {
  label: string;
  gap_status: "strong_gap" | "potential_gap" | "known_limitation" | null;
  highlighted?: boolean;
};

export type GapFlowNode = Node<GapNodeData, "gap">;

export function GapNode({ data }: NodeProps<GapFlowNode>) {
  return (
    <div
      className={`min-w-[200px] max-w-[240px] rounded-[3px] border-2 px-3 py-2 ${
        data.highlighted ? "border-[var(--ink)]" : "border-[var(--ink-soft)]"
      }`}
    >
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">candidate gap</span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] leading-snug text-[var(--ink)]">{data.label}</p>
      {data.gap_status && (
        <span className="eyebrow mt-1 inline-block rounded-[2px] border border-[var(--rule)] px-2 py-0.5 text-[0.625rem] text-[var(--ink-soft)]">
          {GAP_STATUS_LABELS[data.gap_status]}
        </span>
      )}
    </div>
  );
}
