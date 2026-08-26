"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type PaperSummary } from "@/lib/api";

/*
  A keyboard-first jump box (Cmd/Ctrl+K), not a search engine - this app
  already has dedicated search on /corpus (semantic, over abstracts) and
  filters on /assessments and /gaps. This is for getting somewhere fast:
  the seven top-level pages, or a paper by title, without clicking through
  a list first. Fits the instrument identity better than a nav redesign
  would - the whole app already assumes a keyboard-literate operator.
*/

type Destination = { href: string; label: string; group: "go to" };

const DESTINATIONS: Destination[] = [
  { href: "/", label: "assess", group: "go to" },
  { href: "/ask", label: "ask the corpus", group: "go to" },
  { href: "/assessments", label: "assessments", group: "go to" },
  { href: "/corpus", label: "corpus", group: "go to" },
  { href: "/gaps", label: "gap review", group: "go to" },
  { href: "/annotate", label: "annotation workbench", group: "go to" },
  { href: "/admin", label: "pipeline status", group: "go to" },
];

export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    setPapers([]);
    setActiveIndex(0);
  }, []);

  // global Cmd/Ctrl+K to open, Escape to close - registered once, not
  // scoped to any input, so it works no matter what's focused
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((wasOpen) => !wasOpen);
      } else if (e.key === "Escape") {
        close();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [close]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // debounced title search - only once the query looks worth a request
  useEffect(() => {
    const q = query.trim();
    if (q.length < 2) {
      // Clearing stale results once the query drops below the search
      // threshold is intentional - not the accidental-derived-state case
      // this rule otherwise targets.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPapers([]);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      api
        .papers({ q, limit: 6, sort: "date_desc" })
        .then((page) => {
          if (!cancelled) setPapers(page.items);
        })
        .catch(() => {
          if (!cancelled) setPapers([]);
        });
    }, 180);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query]);

  const filteredDestinations = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return DESTINATIONS;
    return DESTINATIONS.filter((d) => d.label.toLowerCase().includes(q));
  }, [query]);

  type Item = { key: string; label: string; meta?: string; go: () => void };

  const items = useMemo<Item[]>(() => {
    const destItems: Item[] = filteredDestinations.map((d) => ({
      key: `dest:${d.href}`,
      label: d.label,
      go: () => router.push(d.href),
    }));
    const paperItems: Item[] = papers.map((p) => ({
      key: `paper:${p.id}`,
      label: p.title,
      meta: p.publication_date?.slice(0, 4) ?? undefined,
      go: () => router.push(`/papers/${p.id}`),
    }));
    return [...destItems, ...paperItems];
  }, [filteredDestinations, papers, router]);

  useEffect(() => {
    // Resetting the highlighted row whenever the result set changes shape
    // is intentional - not the accidental-derived-state case this rule
    // otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setActiveIndex(0);
  }, [items.length, query]);

  function choose(item: Item) {
    item.go();
    close();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = items[activeIndex];
      if (item) choose(item);
    }
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-[var(--ink)]/20 pt-[14vh]" onClick={close}>
      <div
        className="w-full max-w-[32rem] border border-[var(--rule)] bg-[var(--panel)] shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-[var(--rule-soft)] px-4 py-3">
          <span className="readout text-[0.75rem] text-[var(--ink-faint)]">jump to</span>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="a page, or a paper title…"
            className="flex-1 bg-transparent font-[family-name:var(--type-text)] text-[0.9375rem] text-[var(--ink)] placeholder:text-[var(--ink-faint)] focus:outline-none"
          />
          <span className="readout hidden text-[0.6875rem] text-[var(--ink-faint)] sm:inline">esc</span>
        </div>

        <div className="max-h-[22rem] overflow-y-auto py-1">
          {items.length === 0 && (
            <p className="px-4 py-6 text-[0.8125rem] text-[var(--ink-faint)]">
              {query.trim().length >= 2 ? "Nothing matches." : "No pages match yet."}
            </p>
          )}

          {filteredDestinations.length > 0 && (
            <PaletteGroup label="go to">
              {filteredDestinations.map((d) => {
                const index = items.findIndex((it) => it.key === `dest:${d.href}`);
                return (
                  <PaletteRow
                    key={d.href}
                    active={index === activeIndex}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => choose(items[index])}
                  >
                    {d.label}
                  </PaletteRow>
                );
              })}
            </PaletteGroup>
          )}

          {papers.length > 0 && (
            <PaletteGroup label="papers">
              {papers.map((p) => {
                const index = items.findIndex((it) => it.key === `paper:${p.id}`);
                return (
                  <PaletteRow
                    key={p.id}
                    active={index === activeIndex}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => choose(items[index])}
                  >
                    <span className="truncate">{p.title}</span>
                    {p.publication_date && (
                      <span className="readout ml-3 shrink-0 text-[0.6875rem] text-[var(--ink-faint)]">
                        {p.publication_date.slice(0, 4)}
                      </span>
                    )}
                  </PaletteRow>
                );
              })}
            </PaletteGroup>
          )}
        </div>
      </div>
    </div>
  );
}

function PaletteGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="eyebrow block px-4 pt-2 pb-1 text-[0.625rem] text-[var(--ink-faint)]">{label}</span>
      <ul>{children}</ul>
    </div>
  );
}

function PaletteRow({
  active,
  onMouseEnter,
  onClick,
  children,
}: {
  active: boolean;
  onMouseEnter: () => void;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <li>
      <button
        type="button"
        onMouseEnter={onMouseEnter}
        onClick={onClick}
        className={`flex w-full items-center justify-between px-4 py-2 text-left text-[0.875rem] ${
          active ? "bg-[var(--field)] text-[var(--ink)]" : "text-[var(--ink-soft)]"
        }`}
      >
        {children}
      </button>
    </li>
  );
}

/** Small nav-bar affordance that opens the palette - discoverability for
    mouse users, since the shortcut alone is invisible without it. */
export function CommandPaletteTrigger() {
  return (
    <button
      type="button"
      onClick={() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))}
      className="eyebrow flex items-center gap-1.5 text-[var(--ink-faint)] hover:text-[var(--ink)]"
      aria-label="Open the command palette"
    >
      search
      <span className="readout rounded-[2px] border border-[var(--rule)] px-1 py-0.5 text-[0.625rem] leading-none">
        ⌘ + K
      </span>
    </button>
  );
}
