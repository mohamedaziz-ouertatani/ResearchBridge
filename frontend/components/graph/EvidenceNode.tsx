"use client";

import { Handle, Position } from "@xyflow/react";
import type { Node, NodeProps } from "@xyflow/react";

export type EvidenceNodeData = {
  label: string;
  paper_title: string | null;
  section: string | null;
};

export type EvidenceFlowNode = Node<EvidenceNodeData, "evidence">;

export function EvidenceNode({ data }: NodeProps<EvidenceFlowNode>) {
  return (
    <div className="min-w-[200px] max-w-[240px] rounded-[3px] border border-[var(--rule)] px-3 py-2">
      <Handle type="target" position={Position.Top} />
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        evidence{data.section ? ` · ${data.section}` : ""}
      </span>
      <p className="mt-1 line-clamp-3 text-[0.8125rem] italic leading-snug text-[var(--ink-soft)]">
        &quot;{data.label}&quot;
      </p>
      {data.paper_title && (
        <div className="mt-1 truncate text-[0.6875rem] text-[var(--ink-faint)]">{data.paper_title}</div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
