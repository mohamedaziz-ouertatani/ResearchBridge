"use client";

import { useState } from "react";
import { evidenceApi, type EvidenceReviewPayload } from "@/lib/evidenceApi";
import type { EvidenceReview } from "@/lib/assessmentApi";

type EvidenceReviewControlProps = {
  evidenceId: string;
  surfaceType: EvidenceReviewPayload["surface_type"];
  surfaceId: string;
  role?: string;
  initialReview?: EvidenceReview | null;
};

export function EvidenceReviewControl({
  evidenceId,
  surfaceType,
  surfaceId,
  role = "",
  initialReview,
}: EvidenceReviewControlProps) {
  const [review, setReview] = useState(initialReview ?? null);
  const [note, setNote] = useState(initialReview?.note ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);

  async function save(verdict: EvidenceReviewPayload["verdict"]) {
    setBusy(true);
    setError(false);
    try {
      const updated = await evidenceApi.review({
        evidence_id: evidenceId,
        surface_type: surfaceType,
        surface_id: surfaceId,
        role,
        verdict,
        note: note.trim() || null,
      });
      setReview(updated);
    } catch {
      setError(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        review evidence
      </span>
      {(["supported", "unclear", "not_supported"] as const).map((verdict) => (
        <button
          key={verdict}
          type="button"
          disabled={busy}
          onClick={() => void save(verdict)}
          className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.625rem] disabled:opacity-40 ${
            review?.verdict === verdict
              ? "border-[var(--ink)] text-[var(--ink)]"
              : "border-[var(--rule-soft)] text-[var(--ink-faint)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
          }`}
        >
          {verdict.replace("_", " ")}
        </button>
      ))}
      <input
        value={note}
        onChange={(event) => setNote(event.target.value)}
        onBlur={() => review && void save(review.verdict)}
        maxLength={2000}
        placeholder="optional note"
        aria-label="Evidence review note"
        className="min-w-[12rem] border-b border-[var(--rule-soft)] bg-transparent py-1 text-[0.6875rem] text-[var(--ink-soft)] focus:border-[var(--ink)] focus:outline-none"
      />
      {error && (
        <span className="text-[0.6875rem] text-[var(--live)]">save failed</span>
      )}
    </div>
  );
}
