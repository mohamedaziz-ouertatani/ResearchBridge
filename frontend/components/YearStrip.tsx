"use client";

/*
  The corpus's real year distribution, drawn as a calibrated strip.

  It is navigation, not ornament: each bar is a filter control. Showing the
  shape of the corpus up front also makes sampling bias visible - an early
  version of this corpus was 100% one year, and a strip like this would
  have made that obvious immediately.
*/

export function YearStrip({
  byYear,
  activeYear,
  onSelectYear,
}: {
  byYear: Record<string, number>;
  activeYear: number | null;
  onSelectYear: (year: number | null) => void;
}) {
  const years = Object.keys(byYear)
    .map(Number)
    .sort((a, b) => a - b);

  if (years.length === 0) return null;

  const peak = Math.max(...years.map((y) => byYear[String(y)] ?? 0));

  return (
    <div>
      <div className="tickstrip h-14" style={{ ["--cols" as string]: years.length }}>
        {years.map((year) => {
          const count = byYear[String(year)] ?? 0;
          const isActive = activeYear === year;
          return (
            <button
              key={year}
              type="button"
              className="tick"
              data-active={isActive}
              aria-pressed={isActive}
              aria-label={`${count} papers from ${year}${isActive ? " (filter active)" : ""}`}
              onClick={() => onSelectYear(isActive ? null : year)}
            >
              <span
                className="tick-bar"
                style={{ height: `${Math.max((count / peak) * 100, 4)}%` }}
              />
            </button>
          );
        })}
      </div>

      <div
        className="tickstrip mt-2 items-start"
        style={{ ["--cols" as string]: years.length }}
        aria-hidden
      >
        {years.map((year) => (
          <span
            key={year}
            className="readout text-center text-[0.625rem] text-[var(--ink-faint)]"
            style={{ color: activeYear === year ? "var(--near)" : undefined }}
          >
            {String(year).slice(2)}
          </span>
        ))}
      </div>
    </div>
  );
}
