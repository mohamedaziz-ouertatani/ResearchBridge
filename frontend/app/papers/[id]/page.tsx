"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import {
  api,
  type ExtractedClaim,
  type PaperSummary,
  type SearchHit,
} from "@/lib/api";
import { adminApi } from "@/lib/adminApi";
import { CitationGraph } from "@/components/CitationGraph";
import { ExtractedClaims } from "@/components/ExtractedClaims";
import { GaugeLegend } from "@/components/ProximityGauge";
import { PaperRow } from "@/components/PaperRow";
import { SkeletonReport } from "@/components/Skeleton";

function ExcludeToggle({
  paper,
  onChange,
}: {
  paper: PaperSummary;
  onChange: (p: PaperSummary) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const excluded = paper.excluded_at !== null;

  async function toggle() {
    setBusy(true);
    setFailed(false);
    try {
      onChange(await adminApi.excludePaper(paper.id, !excluded));
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="inline-flex items-center gap-2">
      <button
        onClick={toggle}
        disabled={busy}
        className={`eyebrow rounded-[2px] border px-2 py-1 text-[0.6875rem] disabled:opacity-50 ${
          excluded
            ? "border-[var(--live)] text-[var(--live)] hover:opacity-80"
            : "border-[var(--rule)] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)]"
        }`}
      >
        {busy
          ? "saving…"
          : excluded
            ? "excluded — include again"
            : "exclude this paper"}
      </button>
      {failed && (
        <span className="text-[0.6875rem] text-[var(--live)]">save failed</span>
      )}
    </span>
  );
}

export default function PaperDetail({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  const [paper, setPaper] = useState<PaperSummary | null>(null);
  const [claims, setClaims] = useState<ExtractedClaim[]>([]);
  const [similar, setSimilar] = useState<SearchHit[]>([]);
  const [notEmbedded, setNotEmbedded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Clearing stale paper/claims/similar before fetching by `id` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPaper(null);
    setClaims([]);
    setSimilar([]);
    setNotEmbedded(false);
    setError(null);

    api
      .paper(id)
      .then(setPaper)
      .catch(() => setError("That paper isn't in the corpus."));

    api
      .claims(id)
      .then(setClaims)
      .catch(() => {});

    api
      .similar(id, 8)
      .then(setSimilar)
      .catch((e) => {
        // 409 means the paper exists but was never embedded - a real state worth naming
        if (e?.status === 409) setNotEmbedded(true);
      });
  }, [id]);

  if (error) {
    return (
      <main className="mx-auto max-w-[62rem] px-6 py-24 sm:px-8">
        <p className="text-[1.0625rem] text-[var(--ink-soft)]">{error}</p>
        <Link
          href="/corpus"
          className="eyebrow mt-4 inline-block hover:text-[var(--ink)]"
        >
          ← back to the corpus
        </Link>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="border-b border-[var(--rule)] py-5">
        <Link href="/corpus" className="eyebrow hover:text-[var(--ink)]">
          ← corpus
        </Link>
      </header>

      {!paper ? (
        <SkeletonReport />
      ) : (
        <article className="resolve pt-12">
          <div className="mb-5 flex flex-wrap items-center gap-x-3 gap-y-2">
            <span className="readout text-[0.75rem] text-[var(--ink-faint)]">
              {paper.source}:{paper.source_id}
            </span>
            {paper.categories.map((category) => (
              <span
                key={category}
                className="readout rounded-[2px] bg-[var(--field)] px-1.5 py-0.5 text-[0.6875rem] text-[var(--ink-soft)]"
              >
                {category}
              </span>
            ))}
            <ExcludeToggle paper={paper} onChange={setPaper} />
          </div>

          <h1 className="max-w-[34ch] font-[family-name:var(--type-text)] text-[clamp(1.75rem,3.6vw,2.5rem)] leading-[1.15] font-semibold">
            {paper.title}
          </h1>

          <p className="mt-5 max-w-[60ch] font-[family-name:var(--type-display)] text-[0.9375rem] text-[var(--ink-soft)]">
            {paper.authors.length > 0
              ? paper.authors.join(" · ")
              : "Unattributed"}
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-4">
            <span className="readout text-[0.8125rem] text-[var(--ink-faint)]">
              {paper.publication_date ?? "undated"}
            </span>
            {paper.url && (
              <a
                href={paper.url}
                target="_blank"
                rel="noreferrer"
                className="eyebrow underline underline-offset-4 hover:text-[var(--ink)]"
              >
                read online ↗
              </a>
            )}
          </div>

          {paper.abstract && (
            <div className="mt-10 border-t border-[var(--rule)] pt-8">
              <span className="eyebrow">abstract</span>
              <p className="mt-4 max-w-[68ch] text-[1.0625rem] leading-[1.7] text-[var(--ink)]">
                {paper.abstract}
              </p>
            </div>
          )}

          <ExtractedClaims claims={claims} />

          <section className="mt-16">
            <div className="mb-3 flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] pb-3">
              <span className="eyebrow">nearest in meaning</span>
              {similar.length > 0 && <GaugeLegend />}
            </div>

            {notEmbedded ? (
              <p className="py-8 text-[0.9375rem] text-[var(--ink-soft)]">
                This paper hasn&apos;t been embedded yet, so there&apos;s
                nothing to compare it against. Run{" "}
                <code className="readout text-[0.875rem]">rb-embed</code> to
                include it.
              </p>
            ) : (
              <ul>
                {similar.map((hit, i) => (
                  <PaperRow
                    key={hit.paper.id}
                    paper={hit.paper}
                    distance={hit.distance}
                    index={i}
                  />
                ))}
              </ul>
            )}
          </section>

          <CitationGraph paperId={id} />
        </article>
      )}
    </main>
  );
}
