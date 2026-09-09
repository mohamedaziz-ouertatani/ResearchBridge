"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { projectApi, type ResearchProjectSummary } from "@/lib/projectApi";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ResearchProjectSummary[]>([]);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    projectApi
      .list()
      .then((items) => {
        if (!cancelled) setProjects(items);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load research projects.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function createProject(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const created = await projectApi.create(trimmed);
      setProjects((current) => [created, ...current]);
      setName("");
    } catch {
      setError("Couldn't create that project.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />
      <div className="pt-12">
        <span className="eyebrow">research projects</span>
        <h1 className="display mt-3 max-w-[24ch] text-[2rem] leading-tight sm:text-[2.75rem]">
          Keep an idea&apos;s evidence and decisions together.
        </h1>
        <p className="mt-4 max-w-[58ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          A project groups related assessments, notes, and tags. It follows the
          research input, so a rerun appears here as the latest version
          automatically.
        </p>

        <form
          onSubmit={createProject}
          className="mt-8 flex max-w-[34rem] items-end gap-3"
        >
          <label className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
              new project
            </span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={120}
              placeholder="e.g. trustworthy graph learning"
              className="border-b-2 border-[var(--ink)] bg-transparent py-2 font-[family-name:var(--type-text)] text-[1rem] focus:border-[var(--live)] focus:outline-none"
            />
          </label>
          <button
            type="submit"
            disabled={!name.trim() || busy}
            className="eyebrow border border-[var(--ink)] px-3 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:opacity-40"
          >
            {busy ? "creating…" : "create project"}
          </button>
        </form>

        {error && (
          <p className="mt-6 text-[0.875rem] text-[var(--live)]">{error}</p>
        )}
        {loading && (
          <p className="mt-12 text-[0.875rem] text-[var(--ink-faint)]">
            Loading projects…
          </p>
        )}
        {!loading && !error && projects.length === 0 && (
          <p className="mt-12 text-[0.9375rem] text-[var(--ink-soft)]">
            No research projects yet.
          </p>
        )}

        <ul className="mt-10">
          {projects.map((project, index) => (
            <li
              key={project.id}
              className="resolve border-t border-[var(--rule-soft)] py-6 first:border-t-0"
              style={{ animationDelay: `${Math.min(index * 28, 280)}ms` }}
            >
              <Link
                href={`/projects/${project.id}`}
                className="block hover:opacity-80"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <h2 className="font-[family-name:var(--type-text)] text-[1.0625rem] text-[var(--ink)]">
                    {project.name}
                  </h2>
                  <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                    {project.assessment_count} assessment
                    {project.assessment_count === 1 ? "" : "s"}
                  </span>
                </div>
                {project.notes && (
                  <p className="mt-2 max-w-[64ch] text-[0.875rem] leading-relaxed text-[var(--ink-soft)]">
                    {project.notes}
                  </p>
                )}
                {project.tags.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {project.tags.map((tag) => (
                      <span
                        key={tag}
                        className="eyebrow border border-[var(--rule-soft)] px-2 py-1 text-[0.625rem] text-[var(--ink-faint)]"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}
