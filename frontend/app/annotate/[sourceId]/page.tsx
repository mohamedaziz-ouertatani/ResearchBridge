"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useRef, useState } from "react";
import {
  ANNOTATION_FIELDS,
  benchmarkApi,
  type AnnotationDetail,
  type AnnotationSummary,
  type EvidenceItem,
} from "@/lib/benchmarkApi";
import { AnnotationQueue } from "@/components/AnnotationQueue";

type SaveState = "idle" | "saving" | "saved" | "error";

const AUTOSAVE_DELAY_MS = 700;

export default function Workbench({ params }: { params: Promise<{ sourceId: string }> }) {
  const { sourceId } = use(params);

  const [queue, setQueue] = useState<AnnotationSummary[]>([]);
  const [detail, setDetail] = useState<AnnotationDetail | null>(null);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [gap, setGap] = useState({ addressed: "", remaining: "" });
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [selection, setSelection] = useState("");
  const [error, setError] = useState<string | null>(null);

  const dirty = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    benchmarkApi.list().then(setQueue).catch(() => setError("Can't reach the API on port 8000."));
  }, []);

  useEffect(() => {
    dirty.current = false;
    setSaveState("idle");
    setSelection("");

    benchmarkApi
      .detail(sourceId)
      .then((d) => {
        setDetail(d);
        setFields(d.fields);
        setGap(d.research_gap);
        setEvidence(d.key_evidence);
      })
      .catch(() => setError(`No benchmark paper ${sourceId}.`));
  }, [sourceId]);

  const persist = useCallback(async () => {
    setSaveState("saving");
    try {
      const summary = await benchmarkApi.save(sourceId, {
        ...fields,
        research_gap: gap,
        key_evidence: evidence,
      });
      setQueue((rows) => rows.map((r) => (r.source_id === sourceId ? summary : r)));
      setSaveState("saved");
      dirty.current = false;
    } catch {
      setSaveState("error");
    }
  }, [sourceId, fields, gap, evidence]);

  // Autosave: forty papers of work should never hinge on remembering to save.
  useEffect(() => {
    if (!dirty.current) return;
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => void persist(), AUTOSAVE_DELAY_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [fields, gap, evidence, persist]);

  function edit(name: string, value: string) {
    dirty.current = true;
    setFields((f) => ({ ...f, [name]: value }));
  }

  function editGap(key: "addressed" | "remaining", value: string) {
    dirty.current = true;
    setGap((g) => ({ ...g, [key]: value }));
  }

  function captureSelection() {
    const text = window.getSelection()?.toString().trim() ?? "";
    setSelection(text.length > 2 ? text : "");
  }

  function addEvidence() {
    if (!selection) return;
    dirty.current = true;
    setEvidence((items) => [...items, { text: selection, section: "" }]);
    setSelection("");
    window.getSelection()?.removeAllRanges();
  }

  function updateEvidenceSection(index: number, section: string) {
    dirty.current = true;
    setEvidence((items) => items.map((item, i) => (i === index ? { ...item, section } : item)));
  }

  function removeEvidence(index: number) {
    dirty.current = true;
    setEvidence((items) => items.filter((_, i) => i !== index));
  }

  const position = queue.findIndex((p) => p.source_id === sourceId);
  const next = queue[position + 1];
  const previous = queue[position - 1];
  const done = queue.filter((p) => p.is_complete).length;

  if (error) {
    return (
      <main className="mx-auto max-w-[62rem] px-6 py-24">
        <p className="text-[1.0625rem] text-[var(--ink-soft)]">{error}</p>
        <Link href="/" className="eyebrow mt-4 inline-block hover:text-[var(--ink)]">
          ← back to the corpus
        </Link>
      </main>
    );
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] px-6 py-4">
        <div className="flex items-baseline gap-4">
          <Link href="/" className="eyebrow hover:text-[var(--ink)]">
            ← ResearchBridge
          </Link>
          <span className="display text-[0.9375rem]">Annotation workbench</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="eyebrow">
            {done}/{queue.length} papers complete
          </span>
          <span
            className="readout text-[0.6875rem]"
            style={{
              color:
                saveState === "error"
                  ? "var(--live)"
                  : saveState === "saved"
                    ? "var(--near)"
                    : "var(--ink-faint)",
            }}
          >
            {saveState === "saving"
              ? "saving…"
              : saveState === "saved"
                ? "saved"
                : saveState === "error"
                  ? "save failed"
                  : " "}
          </span>
        </div>
      </header>

      <div className="grid flex-1 grid-cols-1 lg:grid-cols-[240px_minmax(0,1fr)_minmax(0,1fr)]">
        {/* worklist */}
        <aside className="border-b border-[var(--rule)] py-2 lg:border-r lg:border-b-0 lg:max-h-[calc(100vh-57px)] lg:overflow-y-auto">
          <AnnotationQueue papers={queue} activeId={sourceId} />
        </aside>

        {/* the paper */}
        <section className="border-b border-[var(--rule)] lg:border-r lg:border-b-0 lg:max-h-[calc(100vh-57px)] lg:overflow-y-auto">
          <div className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-[var(--rule-soft)] bg-[var(--field)] px-5 py-3">
            <span className="eyebrow">the paper</span>
            {selection ? (
              <button
                onClick={addEvidence}
                className="readout rounded-[2px] bg-[var(--near)] px-2 py-1 text-[0.6875rem] text-white"
              >
                add selection as evidence
              </button>
            ) : (
              detail?.url && (
                <a
                  href={detail.url}
                  target="_blank"
                  rel="noreferrer"
                  className="eyebrow underline underline-offset-4 hover:text-[var(--ink)]"
                >
                  on arxiv ↗
                </a>
              )
            )}
          </div>

          <div className="px-5 py-5" onMouseUp={captureSelection}>
            <h1 className="font-[family-name:var(--type-text)] text-[1.25rem] leading-snug font-semibold">
              {detail?.title ?? "…"}
            </h1>
            <p className="readout mt-1 text-[0.6875rem] text-[var(--ink-faint)]">
              {detail?.domain} · {detail?.year} · {sourceId}
            </p>

            {detail && !detail.fulltext ? (
              <p className="mt-6 border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] text-[var(--ink-soft)]">
                No full text cached for this paper. Run{" "}
                <code className="readout text-[0.875rem]">rb-benchmark-fetch</code> to download it.
              </p>
            ) : (
              <pre className="mt-5 font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] whitespace-pre-wrap text-[var(--ink)]">
                {detail?.fulltext}
              </pre>
            )}
          </div>
        </section>

        {/* the annotation */}
        <section className="lg:max-h-[calc(100vh-57px)] lg:overflow-y-auto">
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-[var(--rule-soft)] bg-[var(--field)] px-5 py-3">
            <span className="eyebrow">your annotation</span>
            <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
              {queue[position]?.filled ?? 0}/{queue[position]?.total ?? 10}
            </span>
          </div>

          <div className="space-y-5 px-5 py-5">
            {ANNOTATION_FIELDS.map((field) => (
              <Field
                key={field.name}
                label={field.label}
                prompt={field.prompt}
                value={fields[field.name] ?? ""}
                onChange={(v) => edit(field.name, v)}
              />
            ))}

            <Field
              label="Research gap — addressed"
              prompt="What gap does the paper address?"
              value={gap.addressed}
              onChange={(v) => editGap("addressed", v)}
            />
            <Field
              label="Research gap — remaining"
              prompt="What gap remains?"
              value={gap.remaining}
              onChange={(v) => editGap("remaining", v)}
            />

            <div>
              <span className="eyebrow">key evidence</span>
              <p className="mt-1 text-[0.8125rem] text-[var(--ink-faint)]">
                Select a passage in the paper, then add it here.
              </p>

              {evidence.length === 0 ? (
                <p className="mt-3 text-[0.875rem] text-[var(--ink-soft)]">No passages captured yet.</p>
              ) : (
                <ul className="mt-3 space-y-3">
                  {evidence.map((item, i) => (
                    <li key={i} className="border-l-2 border-[var(--rule)] pl-3">
                      <p className="text-[0.875rem] leading-snug text-[var(--ink)]">“{item.text}”</p>
                      <div className="mt-1.5 flex items-center gap-2">
                        <input
                          value={item.section}
                          onChange={(e) => updateEvidenceSection(i, e.target.value)}
                          placeholder="section, e.g. Discussion"
                          aria-label={`Section for evidence ${i + 1}`}
                          className="readout w-full max-w-[220px] border-b border-[var(--rule)] bg-transparent py-0.5 text-[0.6875rem] focus:border-[var(--ink)] focus:outline-none"
                        />
                        <button
                          onClick={() => removeEvidence(i)}
                          className="eyebrow shrink-0 hover:text-[var(--live)]"
                        >
                          remove
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-[var(--rule)] pt-5">
              {previous ? (
                <Link href={`/annotate/${previous.source_id}`} className="eyebrow hover:text-[var(--ink)]">
                  ← previous
                </Link>
              ) : (
                <span />
              )}
              {next && (
                <Link href={`/annotate/${next.source_id}`} className="eyebrow hover:text-[var(--ink)]">
                  next paper →
                </Link>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function Field({
  label,
  prompt,
  value,
  onChange,
}: {
  label: string;
  prompt: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <span className="mt-0.5 block text-[0.8125rem] text-[var(--ink-faint)]">{prompt}</span>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        className="mt-2 w-full resize-y border border-[var(--rule)] bg-[var(--panel)] px-3 py-2 text-[0.9375rem] leading-relaxed focus:border-[var(--ink)] focus:outline-none"
      />
    </label>
  );
}
