"use client";

import { useEffect, useRef } from "react";

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "confirm",
  cancelLabel = "cancel",
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  cancelLabel?: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    confirmRef.current?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onCancel]);

  if (!open) return null;

  return (
    <div
      role="presentation"
      onClick={onCancel}
      className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--ink)]/40 px-6"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-description"
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-[26rem] border border-[var(--rule)] bg-[var(--panel)] p-6 shadow-lg"
      >
        <span className="eyebrow text-[var(--live)]">confirm</span>
        <h2
          id="confirm-dialog-title"
          className="display mt-3 text-[1.25rem] leading-snug"
        >
          {title}
        </h2>
        <p
          id="confirm-dialog-description"
          className="mt-3 text-[0.875rem] leading-relaxed text-[var(--ink-soft)]"
        >
          {description}
        </p>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="eyebrow px-3 py-2 text-[var(--ink-faint)] hover:text-[var(--ink)] disabled:opacity-40"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="eyebrow border border-[var(--live)] px-3 py-2 text-[var(--live)] hover:bg-[var(--live)] hover:text-[var(--panel)] disabled:opacity-40"
          >
            {busy ? "working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
