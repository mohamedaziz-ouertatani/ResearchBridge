"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { projectApi, type ResearchProjectSummary } from "@/lib/projectApi";

const NEW_PROJECT_VALUE = "__new__";

/** Lets a reader attach the assessment they're looking at to a research
 * project without leaving the report - the reverse direction of
 * app/projects/[id]/page.tsx's "add existing assessment" picker, which
 * only works if you already know which project you want and navigate
 * there first. Mirrors ExportMenu's disclosure pattern (button that opens
 * a small panel, closes on outside click) since both are secondary actions
 * that shouldn't compete with the report content for attention. */
export function AddToProjectControl({
  researchInputId,
}: {
  researchInputId: string;
}) {
  const [open, setOpen] = useState(false);
  const [projects, setProjects] = useState<ResearchProjectSummary[] | null>(
    null,
  );
  const [selected, setSelected] = useState("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addedTo, setAddedTo] = useState<ResearchProjectSummary | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  function toggleOpen() {
    setOpen((v) => !v);
    setError(null);
    if (!projects) {
      projectApi
        .list()
        .then(setProjects)
        .catch(() => setError("Couldn't load your projects."));
    }
  }

  async function add(event: React.FormEvent) {
    event.preventDefault();
    setError(null);

    if (selected === NEW_PROJECT_VALUE) {
      const trimmed = newName.trim();
      if (!trimmed) return;
      setBusy(true);
      try {
        const created = await projectApi.create(trimmed);
        await projectApi.addAssessment(created.id, researchInputId);
        setAddedTo(created);
        setProjects((current) => [created, ...(current ?? [])]);
        setNewName("");
        setSelected("");
      } catch {
        setError("Couldn't create that project.");
      } finally {
        setBusy(false);
      }
      return;
    }

    if (!selected) return;
    const project = projects?.find((p) => p.id === selected);
    setBusy(true);
    try {
      await projectApi.addAssessment(selected, researchInputId);
      if (project) setAddedTo(project);
      setSelected("");
    } catch {
      setError("Couldn't add this to that project.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={toggleOpen}
        aria-expanded={open}
        className="eyebrow flex items-center gap-1.5 rounded-[2px] border border-[var(--rule)] px-3 py-1.5 hover:border-[var(--ink)] hover:text-[var(--ink)]"
      >
        add to project
        <span
          aria-hidden
          className={`text-[0.625rem] transition-transform ${open ? "rotate-180" : ""}`}
        >
          ▾
        </span>
      </button>

      {open && (
        <div className="absolute top-[calc(100%+0.4rem)] right-0 z-10 w-[18rem] border border-[var(--rule)] bg-[var(--panel)] p-4 shadow-lg">
          {addedTo && (
            <p className="mb-3 text-[0.8125rem] leading-relaxed text-[var(--ink-soft)]">
              ✓ added to{" "}
              <Link
                href={`/projects/${addedTo.id}`}
                className="underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
              >
                {addedTo.name}
              </Link>
            </p>
          )}

          {projects === null && !error && (
            <p className="text-[0.8125rem] text-[var(--ink-faint)]">
              Loading projects…
            </p>
          )}

          {projects !== null && (
            <form onSubmit={add} className="flex flex-col gap-3">
              <label className="flex flex-col gap-1">
                <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                  project
                </span>
                <select
                  value={selected}
                  onChange={(event) => setSelected(event.target.value)}
                  disabled={busy}
                  className="border-b border-[var(--ink)] bg-transparent py-2 text-[0.875rem] focus:outline-none disabled:opacity-50"
                >
                  <option value="">choose a project</option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                  <option value={NEW_PROJECT_VALUE}>+ new project…</option>
                </select>
              </label>

              {selected === NEW_PROJECT_VALUE && (
                <input
                  value={newName}
                  onChange={(event) => setNewName(event.target.value)}
                  maxLength={120}
                  placeholder="project name"
                  autoFocus
                  className="border-b border-[var(--rule)] bg-transparent py-2 text-[0.875rem] focus:border-[var(--ink)] focus:outline-none"
                />
              )}

              <button
                type="submit"
                disabled={
                  busy ||
                  !selected ||
                  (selected === NEW_PROJECT_VALUE && !newName.trim())
                }
                className="eyebrow self-start border border-[var(--ink)] px-3 py-1.5 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:opacity-40"
              >
                {busy ? "adding…" : "add"}
              </button>
            </form>
          )}

          {error && (
            <p className="mt-2 text-[0.75rem] text-[var(--live)]">{error}</p>
          )}
        </div>
      )}
    </div>
  );
}
