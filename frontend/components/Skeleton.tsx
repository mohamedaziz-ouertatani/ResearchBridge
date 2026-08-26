/*
  Placeholder shapes shown while a page's first fetch is in flight, sized to
  match the row/report shapes they stand in for so real content doesn't
  jump when it arrives. Flat pulsing bars (see .skeleton-bar in globals.css)
  rather than a shimmer sweep - shimmer is a materials-design signature this
  app's flat instrument surfaces don't otherwise use.
*/

function Bar({ className = "" }: { className?: string }) {
  return <span className={`skeleton-bar block rounded-[2px] ${className}`} />;
}

/** Stands in for a list of PaperRow / AssessmentRow / GapCard items. */
export function SkeletonRows({ count = 4 }: { count?: number }) {
  return (
    <ul>
      {Array.from({ length: count }).map((_, i) => (
        <li
          key={i}
          className="border-t border-[var(--rule-soft)] py-6 first:border-t-0"
          style={{ animationDelay: `${i * 60}ms` }}
        >
          <div className="flex items-center gap-3">
            <Bar className="h-3 w-16" />
            <Bar className="h-3 w-10" />
          </div>
          <Bar className="mt-3 h-4 w-[min(90%,34rem)]" />
          <Bar className="mt-2 h-3 w-40" />
        </li>
      ))}
    </ul>
  );
}

/** Stands in for a single-record page: header block + a few prose fields. */
export function SkeletonReport() {
  return (
    <div className="pt-12">
      <div className="border-b border-[var(--rule)] pb-8">
        <Bar className="h-3 w-24" />
        <Bar className="mt-3 h-9 w-[min(80%,30rem)]" />
      </div>
      {[0, 1, 2].map((i) => (
        <div key={i} className="border-b border-[var(--rule-soft)] py-7" style={{ animationDelay: `${i * 60}ms` }}>
          <Bar className="h-3 w-28" />
          <Bar className="mt-3 h-4 w-full max-w-[58ch]" />
          <Bar className="mt-2 h-4 w-3/4 max-w-[48ch]" />
        </div>
      ))}
    </div>
  );
}

/** Stands in for a row of stat tiles (admin dashboard). */
export function SkeletonStats({ count = 4 }: { count?: number }) {
  return (
    <div className="mt-8 flex flex-wrap gap-x-10 gap-y-6 border-b border-[var(--rule)] pb-8">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{ animationDelay: `${i * 60}ms` }}>
          <Bar className="h-3 w-20" />
          <Bar className="mt-2 h-6 w-14" />
        </div>
      ))}
    </div>
  );
}
