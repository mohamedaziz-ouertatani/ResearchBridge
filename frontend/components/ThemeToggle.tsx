"use client";

import { useEffect, useState } from "react";

/*
  Manual light/dark switch. Defaults to light (the faceplate's original
  look) rather than following prefers-color-scheme - a visitor with a dark
  OS theme shouldn't get a different first impression than one without.
  The choice, once made, is remembered per-browser via localStorage; the
  inline script in layout.tsx applies it before paint so there's no
  light-then-dark flash on load.

  Fixed top-right: bottom-right is claimed by NotificationBell, and
  bottom-left is where Next.js's own dev toolbar sits (dev builds only) -
  top-right is the one corner nothing else in the app uses.
*/

const THEME_KEY = "rb-theme";

function currentTheme(): "light" | "dark" {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  // Reads the attribute the inline script already set, rather than
  // localStorage directly - single source of truth for what's on screen.
  useEffect(() => {
    setTheme(currentTheme());
  }, []);

  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // localStorage unavailable (private mode, quota) - theme still applies for this load
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      className="eyebrow fixed top-5 right-5 z-20 rounded-[2px] border border-[var(--rule)] bg-[var(--panel)] px-3 py-1.5 shadow-lg hover:border-[var(--ink)] hover:text-[var(--ink)]"
    >
      {theme === "dark" ? "light" : "dark"}
    </button>
  );
}
