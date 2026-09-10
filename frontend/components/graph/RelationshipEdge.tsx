"use client";

import { BaseEdge, getBezierPath } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";

const DASH_BY_RELATIONSHIP: Record<string, string | undefined> = {
  supports: undefined,
  contradicts: "6 4",
  contextualizes: "2 4",
  addresses_gap: "10 4",
};

export function RelationshipEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const relationship = (data?.relationship as string | undefined) ?? "supports";
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{ stroke: "var(--ink-soft)", strokeDasharray: DASH_BY_RELATIONSHIP[relationship] }}
    />
  );
}
