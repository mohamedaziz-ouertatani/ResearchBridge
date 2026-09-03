"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { benchmarkApi } from "@/lib/benchmarkApi";

/*
  Entry point to the workbench: jump straight to the first unfinished paper
  rather than making the annotator pick one. Resuming should cost nothing.
*/
export default function AnnotateEntry() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    benchmarkApi
      .list()
      .then((papers) => {
        if (papers.length === 0) {
          setError("No benchmark papers yet. Run rb-benchmark-sample to draw the sample.");
          return;
        }
        const target = papers.find((p) => !p.is_complete) ?? papers[0];
        router.replace(`/annotate/${target.source_id}`);
      })
      .catch(() => setError("Can't reach the API. Is it running on port 8000?"));
  }, [router]);

  return (
    <main className="mx-auto max-w-[80rem] px-6 py-24">
      {error ? (
        <>
          <p className="text-[1.0625rem] text-[var(--ink-soft)]">{error}</p>
          <Link href="/" className="eyebrow mt-4 inline-block hover:text-[var(--ink)]">
            ← back to the corpus
          </Link>
        </>
      ) : (
        <p className="eyebrow">opening the next unfinished paper…</p>
      )}
    </main>
  );
}
