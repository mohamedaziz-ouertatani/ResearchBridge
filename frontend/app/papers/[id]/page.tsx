"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import { api, type ExtractedClaim, type PaperSummary, type SearchHit } from "@/lib/api";
import { ExtractedClaims } from "@/components/ExtractedClaims";
import { GaugeLegend } from "@/components/ProximityGauge";
import { PaperRow } from "@/components/PaperRow";

export default function PaperDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const [paper, setPaper] = useState<PaperSummary | null>(null);
  const [claims, setClaims] = useState<ExtractedClaim[]>([]);
  const [similar, setSimilar] = useState<SearchHit[]>([]);
  const [notEmbedded, setNotEmbedded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPaper(null);
    setClaims([]);
    setSimilar([]);
    setNotEmbedded(false);
    setError(null);

    api.paper(id).then(setPaper).catch(() => setError("That paper isn't in the corpus."));

    api.claims(id).then(setClaims).catch(() => {});

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
        <Link href="/" className="eyebrow mt-4 inline-block hover:text-[var(--ink)]">
          ← back to the corpus
        </Link>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-[62rem] px-6 pb-24 sm:px-8">
      <header className="border-b border-[var(--rule)] py-5">
        <Link href="/" className="eyebrow hover:text-[var(--ink)]">
          ← ResearchBridge
        </Link>
      </header>

      {!paper ? (
        <p className="eyebrow py-16">loading…</p>
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
          </div>

          <h1 className="max-w-[34ch] font-[family-name:var(--type-text)] text-[clamp(1.75rem,3.6vw,2.5rem)] leading-[1.15] font-semibold">
            {paper.title}
          </h1>

          <p className="mt-5 max-w-[60ch] font-[family-name:var(--type-display)] text-[0.9375rem] text-[var(--ink-soft)]">
            {paper.authors.length > 0 ? paper.authors.join(" · ") : "Unattributed"}
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
                read on arxiv ↗
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
                This paper hasn&apos;t been embedded yet, so there&apos;s nothing to compare it against.
                Run <code className="readout text-[0.875rem]">rb-embed</code> to include it.
              </p>
            ) : (
              <ul>
                {similar.map((hit, i) => (
                  <PaperRow key={hit.paper.id} paper={hit.paper} distance={hit.distance} index={i} />
                ))}
              </ul>
            )}
          </section>
        </article>
      )}
    </main>
  );
}
