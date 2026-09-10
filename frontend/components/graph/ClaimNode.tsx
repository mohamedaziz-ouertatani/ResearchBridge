"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";

export type ClaimNodeData = {
  label: string;
  claim_type: string | null;
  confidence: string | null;
  status: string | null;
};

export type ClaimFlowNode = Node<ClaimNodeData, "claim">;

export function ClaimNode({ data }: NodeProps<ClaimFlowNode>) {
  return (
    <div className="min-w-[200px] max-w-[240px] rounded-[3px] border border-[var(--ink)] px-3 py-2">
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        claim{data.claim_type ? ` · ${data.claim_type}` : ""}
      </span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] leading-snug text-[var(--ink)]">{data.label}</p>
      <div className="mt-1 text-[0.6875rem] text-[var(--ink-faint)]">
        {data.confidence ? `${data.confidence} confidence` : null}
        {data.status ? ` · ${data.status}` : null}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
