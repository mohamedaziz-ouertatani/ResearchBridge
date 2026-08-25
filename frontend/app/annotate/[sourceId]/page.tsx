"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useRef, useState } from "react";

import { API_BASE } from "@/lib/api";
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
  const [extractorView, setExtractorView] = useState<"nougat" | "pymupdf">("nougat");

  // mathpix-markdown-it is CommonJS and drags in mathjax-full, opentype.js
  // and highlight.js. Imported statically it breaks this page's chunking
  // (chunks 404 and the component never hydrates), so load it on demand.
  const [nougatHtml, setNougatHtml] = useState("");
  useEffect(() => {
    const mmd = detail?.fulltext_nougat;
    if (!mmd) {
      setNougatHtml("");
      return;
    }
    let cancelled = false;
    import("mathpix-markdown-it").then(({ MathpixMarkdownModel }) => {
      if (cancelled) return;
      // htmlTags stays false: this text is OCR of a third-party PDF, so any
      // raw HTML in it is untrusted and must not reach the DOM. With it off
      // the parser emits only its own markup, which is what makes the
      // dangerouslySetInnerHTML below safe.
      // The backend stores figure links as API-relative paths (it doesn't
      // know this frontend's API_BASE), so the frontend fills that in before
      // rendering rather than the backend baking in a base URL.
      const withImageBase = mmd.replace(/\]\(\/api\/benchmark\//g, `](${API_BASE}/api/benchmark/`);
      setNougatHtml(
        // The text arrives already repaired: normalize_nougat_markdown runs
        // at extraction time, so every consumer - this page, the export, the
        // assessment pipeline - reads the same structurally sound Markdown.
        MathpixMarkdownModel.markdownToHTML(withImageBase, {
          htmlTags: false,
          formulaNumbering: true,
          // MathJax renders to SVG, which carries no text - so selecting an
          // equation yields "" and evidence capture silently drops every
          // variable from a quoted passage. Emitting MathML alongside puts
          // the symbols back in the selection. It does not double up the way
          // KaTeX did: there the HTML layer had text of its own, whereas the
          // SVG here contributes none, so MathML is the only text source.
          outMath: { include_mathml: true },
        }),
      );
    });
    return () => {
      cancelled = true;
    };
  }, [detail?.fulltext_nougat]);

  const dirty = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    benchmarkApi.list().then(setQueue).catch(() => setError("Can't reach the API on port 8000."));
  }, []);

  useEffect(() => {
    dirty.current = false;
    setSaveState("idle");
    setSelection("");
    setExtractorView("nougat");

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
    // MathML puts each token on its own line, so a selection crossing an
    // equation arrives as "...Gossip;\n|\nW\n|\nis the number of...".
    // Collapsing runs of whitespace restores the sentence as it reads on
    // the page, which is what belongs in a quoted piece of evidence.
    const text = window.getSelection()?.toString().replace(/\s+/g, " ").trim() ?? "";
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
    <div className="flex h-screen flex-col">
      <header className="flex flex-none flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] px-6 py-4">
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

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-rows-[minmax(0,1fr)] lg:grid-cols-[240px_minmax(0,1fr)_minmax(0,1fr)]">
        {/* worklist */}
        <aside className="border-b border-[var(--rule)] py-2 lg:h-full lg:overflow-y-auto lg:border-r lg:border-b-0">
          <AnnotationQueue papers={queue} activeId={sourceId} />
        </aside>

        {/* the paper */}
        <section className="border-b border-[var(--rule)] lg:h-full lg:overflow-y-auto lg:border-r lg:border-b-0">
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
                  view source ↗
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

            {detail && !detail.fulltext && !detail.fulltext_nougat ? (
              <p className="mt-6 border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] text-[var(--ink-soft)]">
                No full text cached for this paper. Run{" "}
                <code className="readout text-[0.875rem]">rb-benchmark-fetch</code> to download it.
              </p>
            ) : (
              <>
                {detail?.fulltext_nougat && detail?.fulltext && (
                  <div className="mb-3 flex gap-1">
                    <button
                      onClick={() => setExtractorView("nougat")}
                      className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
                        extractorView === "nougat"
                          ? "border-[var(--ink)] text-[var(--ink)]"
                          : "border-[var(--rule)] text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
                      }`}
                    >
                      Nougat
                    </button>
                    <button
                      onClick={() => setExtractorView("pymupdf")}
                      className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] ${
                        extractorView === "pymupdf"
                          ? "border-[var(--ink)] text-[var(--ink)]"
                          : "border-[var(--rule)] text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
                      }`}
                    >
                      PyMuPDF
                    </button>
                  </div>
                )}

                {(extractorView === "nougat" ? detail?.fulltext_nougat : detail?.fulltext) ? (
                  extractorView === "nougat" && detail?.fulltext_nougat ? (
                    <div
                      className="nougat-mmd mt-5 max-w-none font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] text-[var(--ink)]"
                      dangerouslySetInnerHTML={{ __html: nougatHtml }}
                    />
                  ) : (
                    <pre className="mt-5 font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] whitespace-pre-wrap text-[var(--ink)]">
                      {detail?.fulltext}
                    </pre>
                  )
                ) : (
                  <pre className="mt-5 font-[family-name:var(--type-text)] text-[0.9375rem] leading-[1.65] whitespace-pre-wrap text-[var(--ink)]">
                    {detail?.fulltext}
                  </pre>
                )}
              </>
            )}
          </div>
        </section>

        {/* the annotation */}
        <section className="lg:h-full lg:overflow-y-auto">
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
