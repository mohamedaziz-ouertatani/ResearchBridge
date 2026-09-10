"use client";

import { useEffect, useState } from "react";
import { assessmentApi, type GapDensityBucket } from "@/lib/assessmentApi";

export type GapDensityOverlayProps = {
  assessmentId: string;
  gapCategories: string[];
  onHighlightChange: (highlighted: boolean) => void;
};

export function GapDensityOverlay({ assessmentId, gapCategories, onHighlightChange }: GapDensityOverlayProps) {
  const [buckets, setBuckets] = useState<GapDensityBucket[] | null>(null);
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    assessmentApi.gapDensity(assessmentId).then((data) => setBuckets(data.buckets));
  }, [assessmentId]);

  useEffect(() => {
    if (!enabled || !buckets) {
      onHighlightChange(false);
      return;
    }
    const average = buckets.reduce((sum, b) => sum + b.gap_count, 0) / (buckets.length || 1);
    const highlighted = buckets.some((b) => gapCategories.includes(b.category) && b.gap_count >= average);
    onHighlightChange(highlighted);
  }, [enabled, buckets, gapCategories, onHighlightChange]);

  return (
    <button
      type="button"
      onClick={() => setEnabled((v) => !v)}
      disabled={!buckets}
      className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
        enabled ? "border-[var(--ink)] text-[var(--ink)]" : "border-[var(--rule)] text-[var(--ink-faint)]"
      }`}
    >
      gap density overlay
    </button>
  );
}
