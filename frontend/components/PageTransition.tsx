"use client";

import { usePathname } from "next/navigation";

/*
  Keying on pathname remounts this wrapper on every route change, replaying
  the same .resolve fade-up that report/list rows already use on mount - so
  navigating between pages reads as one continuous instrument rather than a
  hard cut, without introducing a second animation vocabulary.
*/
export function PageTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div key={pathname} className="resolve">
      {children}
    </div>
  );
}
