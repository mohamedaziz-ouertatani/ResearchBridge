"use client";

import { useEffect, useRef, useState } from "react";
import { adminApi, type Notification } from "@/lib/adminApi";

/*
  Global notification bell: pipeline runs finishing/failing, plus the
  assessment and candidate-gap review queues, all visible from any page
  without visiting /admin. Fixed bottom-right rather than folded into each
  page's own header - every page in this app has a differently-shaped
  header (nav links, save state, paper progress...), so a shared header
  slot would mean editing all of them and fighting for space in a crowded
  top-right corner. Bottom-right is unclaimed on every page.

  Read/unread state is entirely client-side (a set of seen notification
  ids in localStorage) - there's no server-side "user" to track it against,
  and the backend already keeps ids stable per event (see Notification's
  docstring in schemas.py), which is exactly what a client-tracked seen-set
  needs.
*/

const SEEN_KEY = "rb-notifications-seen";
const POLL_MS = 20000;
const MAX_SEEN_IDS = 300;

function loadSeen(): Set<string> {
  try {
    const raw = localStorage.getItem(SEEN_KEY);
    return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
  } catch {
    return new Set();
  }
}

function saveSeen(ids: Set<string>) {
  try {
    localStorage.setItem(SEEN_KEY, JSON.stringify(Array.from(ids).slice(-MAX_SEEN_IDS)));
  } catch {
    // localStorage unavailable (private mode, quota) - degrades to "everything always unread"
  }
}

export function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [seenIds, setSeenIds] = useState<Set<string>>(() => loadSeen());
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    function poll() {
      adminApi
        .notifications()
        .then((next) => {
          if (!cancelled) setNotifications(next);
        })
        .catch(() => {
          // API unreachable - keep showing the last-known list rather than blanking it.
        });
    }
    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const unreadCount = notifications.filter((n) => !seenIds.has(n.id)).length;

  function toggle() {
    setOpen((wasOpen) => {
      const nowOpen = !wasOpen;
      if (nowOpen && notifications.length > 0) {
        const merged = new Set(seenIds);
        for (const n of notifications) merged.add(n.id);
        setSeenIds(merged);
        saveSeen(merged);
      }
      return nowOpen;
    });
  }

  return (
    <div ref={containerRef} className="fixed right-4 bottom-4 z-50">
      {open && (
        <div className="mb-2 max-h-[28rem] w-[22rem] overflow-y-auto border border-[var(--rule)] bg-[var(--panel)] shadow-lg">
          <div className="sticky top-0 border-b border-[var(--rule-soft)] bg-[var(--panel)] px-3 py-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">notifications</span>
          </div>
          {notifications.length === 0 ? (
            <p className="px-3 py-6 text-center text-[0.8125rem] text-[var(--ink-faint)]">Nothing yet.</p>
          ) : (
            <ul>
              {notifications.map((n) => (
                <li key={n.id} className="border-b border-[var(--rule-soft)] px-3 py-2.5 last:border-b-0">
                  <p
                    className={`line-clamp-3 [overflow-wrap:anywhere] text-[0.8125rem] leading-snug ${
                      n.severity === "error" ? "text-[var(--live)]" : "text-[var(--ink)]"
                    }`}
                    title={n.message}
                  >
                    {n.message}
                  </p>
                  <span className="readout mt-1 block text-[0.625rem] text-[var(--ink-faint)]">
                    {new Date(n.created_at).toLocaleString()}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <button
        onClick={toggle}
        aria-label={unreadCount > 0 ? `Notifications, ${unreadCount} unread` : "Notifications"}
        className="relative flex h-9 w-9 items-center justify-center rounded-[2px] border border-[var(--rule)] bg-[var(--panel)] text-[var(--ink-soft)] shadow-sm hover:border-[var(--ink)] hover:text-[var(--ink)]"
      >
        <BellIcon />
        {unreadCount > 0 && (
          <span className="readout absolute -top-1.5 -right-1.5 flex h-4 min-w-[1rem] items-center justify-center rounded-full bg-[var(--live)] px-1 text-[0.625rem] text-white">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>
    </div>
  );
}

function BellIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path
        d="M8 1.5C6 1.5 4.5 3 4.5 5v2.2c0 .5-.2 1-.5 1.4L3 9.8c-.4.5 0 1.2.6 1.2h8.8c.6 0 1-.7.6-1.2l-1-1.2c-.3-.4-.5-.9-.5-1.4V5c0-2-1.5-3.5-3.5-3.5Z"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinejoin="round"
      />
      <path d="M6.5 13a1.5 1.5 0 0 0 3 0" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
    </svg>
  );
}
