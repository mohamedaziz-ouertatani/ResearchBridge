"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { CommandPaletteTrigger } from "@/components/CommandPalette";

/*
  One shared header for every top-level page, replacing what used to be a
  bespoke <header> per page (some with the full link set, most with only a
  "← ResearchBridge" back-link and nothing else). Record-detail pages
  (assessments/[id], annotate/[sourceId], papers/[id]) intentionally keep
  their own simpler back-link header - they're a focused single-record
  view, not a place to jump elsewhere from.

  One row, not a wrapping or stacked one: the assess-and-review loop most
  visits are for stays as direct links, and the four corpus-infrastructure
  pages - visited far less often, by operators rather than regular users -
  fold behind a single "infrastructure" disclosure. That keeps the row's
  width fixed regardless of viewport instead of growing with every page
  the app gains. Below `sm` the whole row collapses into one toggle.
*/

const PRIMARY = [
  { href: "/", label: "assess" },
  { href: "/ask", label: "ask the corpus" },
  { href: "/assessments", label: "assessments" },
];

const INFRASTRUCTURE = [
  { href: "/corpus", label: "corpus" },
  { href: "/trends", label: "trends" },
  { href: "/gaps", label: "gap review" },
  { href: "/annotate", label: "annotation workbench" },
  { href: "/admin", label: "pipeline status" },
];

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  if (href === "/corpus") return pathname === "/corpus" || pathname.startsWith("/papers");
  return pathname.startsWith(href);
}

function NavLink({ href, label, active }: { href: string; label: string; active: boolean }) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`eyebrow ${active ? "text-[var(--ink)]" : "text-[var(--ink-faint)] hover:text-[var(--ink)]"}`}
    >
      {label}
    </Link>
  );
}

/** The four corpus-infrastructure pages, tucked behind one disclosure -
    same panel treatment as NotificationBell's dropdown (border/panel/
    shadow-lg), so the two fixed overlays in this app read as one system. */
function InfrastructureMenu({ pathname }: { pathname: string }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const active = INFRASTRUCTURE.some((item) => isActive(pathname, item.href));

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={`eyebrow flex items-center gap-1 ${active ? "text-[var(--ink)]" : "text-[var(--ink-faint)] hover:text-[var(--ink)]"}`}
      >
        infrastructure
        <span aria-hidden className={`text-[0.625rem] transition-transform ${open ? "rotate-180" : ""}`}>
          ▾
        </span>
      </button>

      {open && (
        <div className="absolute top-[calc(100%+0.6rem)] left-1/2 z-10 w-[13rem] -translate-x-1/2 border border-[var(--rule)] bg-[var(--panel)] py-1 shadow-lg">
          {INFRASTRUCTURE.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive(pathname, item.href) ? "page" : undefined}
              onClick={() => setOpen(false)}
              className={`block px-3 py-2 text-[0.8125rem] ${
                isActive(pathname, item.href)
                  ? "text-[var(--ink)]"
                  : "text-[var(--ink-soft)] hover:bg-[var(--field)] hover:text-[var(--ink)]"
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export function Nav() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  // A route change always means navigation happened - closing whatever was
  // open is intentional, not the accidental-derived-state case this rule
  // otherwise targets.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMenuOpen(false);
  }, [pathname]);

  return (
    <header className="border-b border-[var(--rule)] py-5">
      <div className="flex items-center justify-between gap-4">
        <Link href="/" className="display shrink-0 text-[1.0625rem] hover:text-[var(--ink-soft)]">
          ResearchBridge
        </Link>

        {/* one row: direct links to the core loop, infrastructure folded
            behind a disclosure, fixed width regardless of viewport */}
        <div className="hidden items-center gap-x-6 sm:flex">
          <nav className="flex items-baseline gap-x-5">
            {PRIMARY.map((item) => (
              <NavLink key={item.href} {...item} active={isActive(pathname, item.href)} />
            ))}
            <InfrastructureMenu pathname={pathname} />
          </nav>
          <span aria-hidden className="h-3 w-px bg-[var(--rule)]" />
          <CommandPaletteTrigger />
        </div>

        {/* narrow viewports: a single toggle instead of a squeezed row */}
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-expanded={menuOpen}
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          className="eyebrow shrink-0 rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)] sm:hidden"
        >
          {menuOpen ? "close" : "menu"}
        </button>
      </div>

      {menuOpen && (
        <nav className="mt-4 flex flex-col gap-1 border-t border-[var(--rule-soft)] pt-4 sm:hidden">
          {PRIMARY.map((item) => (
            <MobileNavLink key={item.href} {...item} active={isActive(pathname, item.href)} />
          ))}
          <span aria-hidden className="my-2 h-px bg-[var(--rule-soft)]" />
          {INFRASTRUCTURE.map((item) => (
            <MobileNavLink key={item.href} {...item} active={isActive(pathname, item.href)} />
          ))}
          <span aria-hidden className="my-2 h-px bg-[var(--rule-soft)]" />
          <div className="py-2">
            <CommandPaletteTrigger />
          </div>
        </nav>
      )}
    </header>
  );
}

function MobileNavLink({ href, label, active }: { href: string; label: string; active: boolean }) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`eyebrow py-2 ${active ? "text-[var(--ink)]" : "text-[var(--ink-faint)]"}`}
    >
      {label}
    </Link>
  );
}
