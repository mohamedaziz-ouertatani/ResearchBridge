"use client";

import { useEffect, useRef } from "react";

/** An in-app styled stand-in for window.confirm(), for destructive actions
 * (delete project, delete collection) where the OS-native popup breaks the
 * app's own visual language. Same overlay/panel shape as CommandPalette -
 * fixed backdrop, centered bordered panel, click-outside and Escape both
 * dismiss like cancelling. */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "delete",
  cancelLabel = "cancel",
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  description?: string;
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
      className="fixed inset-0 z-[60] flex items-center justify-center bg-[var(--ink)]/20 px-6"
      onClick={onCancel}
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        className="w-full max-w-[26rem] border border-[var(--rule)] bg-[var(--panel)] p-6 shadow-lg"
        onClick={(event) => event.stopPropagation()}
      >
        <h2
          id="confirm-dialog-title"
          className="display text-[1.0625rem] text-[var(--ink)]"
        >
          {title}
        </h2>
        {description && (
          <p className="mt-2 text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
            {description}
          </p>
        )}
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="eyebrow px-3 py-1.5 text-[var(--ink-soft)] hover:text-[var(--ink)] disabled:opacity-40"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="eyebrow border border-[var(--live)] px-3 py-1.5 text-[var(--live)] hover:bg-[var(--live)] hover:text-[var(--panel)] disabled:opacity-40"
          >
            {busy ? "working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
