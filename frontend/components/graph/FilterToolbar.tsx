"use client";

export type KindFilter = { claim: boolean; evidence: boolean; gap: boolean };
export type ConfidenceFilter = { high: boolean; medium: boolean; low: boolean };

export type FilterToolbarProps = {
  kinds: KindFilter;
  onKindsChange: (kinds: KindFilter) => void;
  confidence: ConfidenceFilter;
  onConfidenceChange: (confidence: ConfidenceFilter) => void;
};

const KIND_LABELS: Record<keyof KindFilter, string> = { claim: "claims", evidence: "evidence", gap: "gaps" };
const CONFIDENCE_LABELS: Record<keyof ConfidenceFilter, string> = { high: "high", medium: "medium", low: "low" };

export function FilterToolbar({ kinds, onKindsChange, confidence, onConfidenceChange }: FilterToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-[var(--rule-soft)] pb-3">
      <div className="flex items-center gap-2">
        {(Object.keys(KIND_LABELS) as (keyof KindFilter)[]).map((kind) => (
          <label key={kind} className="eyebrow flex items-center gap-1 text-[0.6875rem] text-[var(--ink-soft)]">
            <input
              type="checkbox"
              checked={kinds[kind]}
              onChange={(e) => onKindsChange({ ...kinds, [kind]: e.target.checked })}
            />
            {KIND_LABELS[kind]}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2">
        {(Object.keys(CONFIDENCE_LABELS) as (keyof ConfidenceFilter)[]).map((level) => (
          <label key={level} className="eyebrow flex items-center gap-1 text-[0.6875rem] text-[var(--ink-soft)]">
            <input
              type="checkbox"
              checked={confidence[level]}
              onChange={(e) => onConfidenceChange({ ...confidence, [level]: e.target.checked })}
            />
            {CONFIDENCE_LABELS[level]}
          </label>
        ))}
      </div>
    </div>
  );
}
