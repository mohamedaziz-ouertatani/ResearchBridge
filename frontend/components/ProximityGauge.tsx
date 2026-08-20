/*
  The signature element.

  Every hit carries a cosine distance, and printing it as a bare number
  ("0.531") tells you nothing on its own. Drawn on a shared calibrated
  scale, the whole result set becomes readable at a glance: you can see
  where relevance falls off a cliff, and whether the top hit is genuinely
  close or merely the closest of a bad lot.

  The scale is fixed (not normalized per result set) so the reading means
  the same thing on every query. Real hits land roughly 0.35-0.85.
*/

const SCALE_MAX = 1;
const TICKS = [0.25, 0.5, 0.75];

export function ProximityGauge({ distance }: { distance: number }) {
  const clamped = Math.min(Math.max(distance, 0), SCALE_MAX);
  const position = (clamped / SCALE_MAX) * 100;
  // saturated teal when near, washed out when far
  const color = `color-mix(in oklab, var(--near), var(--far) ${position}%)`;

  return (
    <div className="flex items-center gap-3">
      <span className="readout text-[0.8125rem] tabular-nums" style={{ color }}>
        {distance.toFixed(3)}
      </span>

      <div
        className="relative h-[18px] w-full min-w-[80px] max-w-[190px] flex-1"
        role="img"
        aria-label={`Semantic distance ${distance.toFixed(3)} of a possible ${SCALE_MAX.toFixed(1)}; lower is closer in meaning`}
      >
        {/* track */}
        <div className="absolute top-1/2 h-px w-full -translate-y-1/2 bg-[var(--rule)]" />

        {/* calibration ticks */}
        {TICKS.map((tick) => (
          <div
            key={tick}
            className="absolute top-1/2 h-[5px] w-px -translate-y-1/2 bg-[var(--rule)]"
            style={{ left: `${(tick / SCALE_MAX) * 100}%` }}
          />
        ))}

        {/* end caps */}
        <div className="absolute top-1/2 left-0 h-[9px] w-px -translate-y-1/2 bg-[var(--ink-faint)]" />
        <div className="absolute top-1/2 right-0 h-[9px] w-px -translate-y-1/2 bg-[var(--ink-faint)]" />

        {/* needle */}
        <div
          className="absolute top-1/2 h-[13px] w-[2px] -translate-x-1/2 -translate-y-1/2 transition-[left] duration-300"
          style={{ left: `${position}%`, background: color }}
        />
      </div>
    </div>
  );
}

/** Legend for the gauge - shown once above a result set, not per row. */
export function GaugeLegend() {
  return (
    <div className="flex items-center gap-2">
      <span className="eyebrow">closer</span>
      <div
        className="h-[3px] w-16"
        style={{ background: "linear-gradient(to right, var(--near), var(--far))" }}
      />
      <span className="eyebrow">further</span>
    </div>
  );
}
