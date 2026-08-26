"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/*
  One shared header for every top-level page, replacing what used to be a
  bespoke <header> per page (some with the full link set, most with only a
  "← ResearchBridge" back-link and nothing else). Grouped into two clusters
  rather than one flat row: the assess-and-review loop a regular user works
  in, and the corpus infrastructure that supports it (matches the grouping
  already described in page.tsx's "what you get back" copy). Record-detail
  pages (assessments/[id], annotate/[sourceId], papers/[id]) intentionally
  keep their own simpler back-link header - they're a focused single-record
  view, not a place to jump elsewhere from.
*/

const PRIMARY = [
  { href: "/", label: "assess" },
  { href: "/ask", label: "ask the corpus" },
  { href: "/assessments", label: "assessments" },
];

const INFRASTRUCTURE = [
  { href: "/corpus", label: "corpus" },
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

export function Nav() {
  const pathname = usePathname();

  return (
    <header className="flex flex-wrap items-center justify-between gap-x-8 gap-y-3 border-b border-[var(--rule)] py-5">
      <Link href="/" className="display shrink-0 text-[1.0625rem] hover:text-[var(--ink-soft)]">
        ResearchBridge
      </Link>
      <nav className="flex flex-wrap items-baseline gap-x-8 gap-y-2">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2">
          {PRIMARY.map((item) => (
            <NavLink key={item.href} {...item} active={isActive(pathname, item.href)} />
          ))}
        </div>
        <span aria-hidden className="hidden h-3 w-px self-center bg-[var(--rule)] sm:block" />
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2">
          {INFRASTRUCTURE.map((item) => (
            <NavLink key={item.href} {...item} active={isActive(pathname, item.href)} />
          ))}
        </div>
      </nav>
    </header>
  );
}
