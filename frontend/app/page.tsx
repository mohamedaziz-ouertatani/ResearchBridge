"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { assessmentApi } from "@/lib/assessmentApi";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Nav } from "@/components/Nav";

/*
  The console: where a research idea or paper goes in.

  This is the product's core loop (blueprint Sec 2A) - the corpus explorer,
  annotation workbench, and gap review are the infrastructure that makes it
  possible, so they sit in the nav rather than on the landing surface.
*/

export default function Console() {
  const router = useRouter();
  const [idea, setIdea] = useState("");
  const [busy, setBusy] = useState<"idea" | "file" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function run(work: () => Promise<{ id: string }>, kind: "idea" | "file") {
    setBusy(kind);
    setError(null);
    try {
      const assessment = await work();
      router.push(`/assessments/${assessment.id}`);
    } catch {
      setError(
        "The assessment didn't run. Check the API is up on port 8000 — the embedding model can take a moment to load on first use.",
      );
      setBusy(null);
    }
  }

  function submitIdea(event: React.FormEvent) {
    event.preventDefault();
    const text = idea.trim();
    if (!text) return;
    void run(() => assessmentApi.create(text), "idea");
  }

  function submitFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    void run(() => assessmentApi.upload(file), "file");
  }

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <section className="pt-16 pb-10">
        <h1 className="display max-w-[18ch] text-[clamp(2.25rem,5.5vw,3.5rem)]">
          Check an idea against the literature.
        </h1>
        <p className="mt-4 max-w-[54ch] text-[1.0625rem] leading-relaxed text-[var(--ink-soft)]">
          Describe a research idea or upload a paper. ResearchBridge searches the corpus for the
          most related work, compares your input against it passage by passage, and reports what
          is already solved, what gap remains, and how much of that the literature actually
          supports.
        </p>
        <p className="mt-3 max-w-[54ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Nothing is generated from scratch: every claim in the resulting report is grounded in a
          real passage from a real paper, so you can trace each finding back to where it came
          from. If the corpus doesn&apos;t say something, the report leaves it unassessed rather
          than guessing.
        </p>

        <form onSubmit={submitIdea} className="mt-10">
          <label htmlFor="idea" className="eyebrow inline-flex items-center gap-1.5">
            research idea
            <InfoTooltip text="Type a short description of an idea you're considering. ResearchBridge treats this text the same way it treats an uploaded paper — as the thing to check against the literature." />
          </label>
          <textarea
            id="idea"
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            rows={4}
            disabled={busy !== null}
            placeholder="e.g. real-time fraud detection using graph transformers"
            className="mt-3 w-full resize-y border-b-2 border-[var(--ink)] bg-transparent py-2 font-[family-name:var(--type-text)] text-[1.0625rem] leading-relaxed placeholder:text-[var(--ink-faint)] focus:border-[var(--live)] focus:outline-none disabled:opacity-50"
          />

          <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3">
            <button
              type="submit"
              disabled={!idea.trim() || busy !== null}
              className="eyebrow rounded-[2px] border border-[var(--ink)] px-4 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:border-[var(--rule)] disabled:text-[var(--ink-faint)] disabled:hover:bg-transparent disabled:hover:text-[var(--ink-faint)]"
            >
              {busy === "idea" ? "assessing…" : "assess this idea"}
            </button>

            <span className="eyebrow text-[var(--ink-faint)]">or</span>

            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={busy !== null}
              className="eyebrow underline underline-offset-4 hover:text-[var(--ink)] disabled:text-[var(--ink-faint)]"
            >
              {busy === "file" ? "reading document…" : "upload a paper (pdf or text)"}
            </button>
            <InfoTooltip text="Have a draft or a full paper instead of a short description? Upload it and ResearchBridge assesses the document itself — same report, same grounding, no need to summarize it yourself first." />
            <input
              ref={fileRef}
              type="file"
              accept=".pdf,.txt,.md,text/plain,application/pdf"
              onChange={submitFile}
              className="hidden"
            />
          </div>
        </form>

        {busy && (
          <p className="eyebrow mt-8">
            retrieving related work · comparing · grounding each finding
          </p>
        )}

        {error && (
          <p className="mt-8 max-w-[58ch] border-l-2 border-[var(--live)] pl-4 text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
            {error}
          </p>
        )}
      </section>

      <section className="border-t border-[var(--rule)] pt-8">
        <span className="eyebrow">what you get back</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          A report covering related research, existing solutions, novelty, the remaining research
          gap, applications, technical feasibility, risks, and what still needs outside validation.
          Every finding shows the passages it came from, and anything the literature could not
          support is left unassessed rather than filled in.
        </p>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          Every past assessment is saved and can be reopened or marked reviewed from the{" "}
          <Link href="/assessments" className="underline hover:text-[var(--ink)]">
            assessments
          </Link>{" "}
          page. The other pages in the nav above are the infrastructure behind that report: the{" "}
          <Link href="/corpus" className="underline hover:text-[var(--ink)]">
            corpus
          </Link>{" "}
          is the paper collection itself, the{" "}
          <Link href="/gaps" className="underline hover:text-[var(--ink)]">
            gap review
          </Link>{" "}
          queue is where cross-paper research gaps get human-approved, the{" "}
          <Link href="/annotate" className="underline hover:text-[var(--ink)]">
            annotation workbench
          </Link>{" "}
          is where a person labels a benchmark sample for evaluating extraction quality, and{" "}
          <Link href="/admin" className="underline hover:text-[var(--ink)]">
            pipeline status
          </Link>{" "}
          runs and monitors the ingestion/extraction/embedding jobs that build the corpus in the
          first place.
        </p>
      </section>
    </main>
  );
}
