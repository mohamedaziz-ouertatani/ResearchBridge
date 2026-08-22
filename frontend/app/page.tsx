"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { assessmentApi } from "@/lib/assessmentApi";

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
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] py-5">
        <span className="display text-[1.0625rem]">ResearchBridge</span>
        <div className="flex items-baseline gap-5">
          <Link href="/assessments" className="eyebrow hover:text-[var(--ink)]">
            assessments →
          </Link>
          <Link href="/corpus" className="eyebrow hover:text-[var(--ink)]">
            corpus →
          </Link>
          <Link href="/gaps" className="eyebrow hover:text-[var(--ink)]">
            gap review →
          </Link>
          <Link href="/annotate" className="eyebrow hover:text-[var(--ink)]">
            annotation workbench →
          </Link>
          <Link href="/admin" className="eyebrow hover:text-[var(--ink)]">
            pipeline status →
          </Link>
        </div>
      </header>

      <section className="pt-16 pb-10">
        <h1 className="display max-w-[18ch] text-[clamp(2.25rem,5.5vw,3.5rem)]">
          Check an idea against the literature.
        </h1>
        <p className="mt-4 max-w-[54ch] text-[1.0625rem] leading-relaxed text-[var(--ink-soft)]">
          Describe a research idea or upload a paper. ResearchBridge finds the related work,
          compares your input against it, and reports what is already solved, what gap remains, and
          how much of that the literature actually supports.
        </p>

        <form onSubmit={submitIdea} className="mt-10">
          <label htmlFor="idea" className="eyebrow">
            research idea
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
      </section>
    </main>
  );
}
