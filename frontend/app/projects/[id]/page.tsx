"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useState } from "react";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Nav } from "@/components/Nav";
import { assessmentApi, type AssessmentSummary } from "@/lib/assessmentApi";
import { projectApi, type ResearchProject } from "@/lib/projectApi";

export default function ProjectPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const [project, setProject] = useState<ResearchProject | null>(null);
  const [available, setAvailable] = useState<AssessmentSummary[]>([]);
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [tags, setTags] = useState("");
  const [selectedInputId, setSelectedInputId] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  async function load() {
    const [loaded, assessmentPage] = await Promise.all([
      projectApi.get(id),
      assessmentApi.list("all"),
    ]);
    setProject(loaded);
    setName(loaded.name);
    setNotes(loaded.notes ?? "");
    setTags(loaded.tags.join(", "));
    const attached = new Set(
      loaded.assessments.map((assessment) => assessment.research_input_id),
    );
    setAvailable(
      assessmentPage.items.filter(
        (assessment) => !attached.has(assessment.research_input_id),
      ),
    );
  }

  useEffect(() => {
    let cancelled = false;
    // Reset the detail view when the route id changes; the fetch below owns
    // the next value.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    setError(null);
    Promise.all([projectApi.get(id), assessmentApi.list("all")])
      .then(([loaded, assessmentPage]) => {
        if (cancelled) return;
        setProject(loaded);
        setName(loaded.name);
        setNotes(loaded.notes ?? "");
        setTags(loaded.tags.join(", "));
        const attached = new Set(
          loaded.assessments.map((assessment) => assessment.research_input_id),
        );
        setAvailable(
          assessmentPage.items.filter(
            (assessment) => !attached.has(assessment.research_input_id),
          ),
        );
      })
      .catch(() => {
        if (!cancelled) setError("That research project could not be loaded.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function saveProject(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await projectApi.update(id, {
        name: name.trim(),
        notes: notes.trim() || null,
        tags: tags
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
      });
      await load();
    } catch {
      setError("Couldn't save the project.");
    } finally {
      setBusy(false);
    }
  }

  async function addAssessment(event: React.FormEvent) {
    event.preventDefault();
    if (!selectedInputId) return;
    setBusy(true);
    try {
      await projectApi.addAssessment(id, selectedInputId);
      setSelectedInputId("");
      await load();
    } catch {
      setError("Couldn't add that assessment.");
    } finally {
      setBusy(false);
    }
  }

  async function removeAssessment(inputId: string) {
    setBusy(true);
    try {
      await projectApi.removeAssessment(id, inputId);
      await load();
    } catch {
      setError("Couldn't remove that assessment.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteProject() {
    setConfirmingDelete(false);
    setBusy(true);
    try {
      await projectApi.remove(id);
      router.push("/projects");
    } catch {
      setError("Couldn't delete the project.");
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />
      <div className="pt-12">
        {error && (
          <p className="mb-6 border-l-2 border-[var(--live)] pl-4 text-[0.875rem] text-[var(--ink-soft)]">
            {error}
          </p>
        )}
        {loading && (
          <p className="text-[0.875rem] text-[var(--ink-faint)]">
            Loading project…
          </p>
        )}
        {!loading && project && (
          <>
            <div className="flex flex-wrap items-baseline justify-between gap-4">
              <div>
                <span className="eyebrow">research project</span>
                <h1 className="display mt-3 max-w-[24ch] text-[2rem] leading-tight sm:text-[2.75rem]">
                  {project.name}
                </h1>
              </div>
              <Link
                href="/projects"
                className="eyebrow text-[var(--ink-soft)] hover:text-[var(--ink)]"
              >
                ← all projects
              </Link>
            </div>

            <form
              onSubmit={saveProject}
              className="mt-10 grid gap-5 border-y border-[var(--rule)] py-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]"
            >
              <label className="flex flex-col gap-2">
                <span className="eyebrow">project name</span>
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  maxLength={120}
                  className="border-b border-[var(--ink)] bg-transparent py-2 text-[0.9375rem] focus:outline-none"
                />
              </label>
              <label className="flex flex-col gap-2">
                <span className="eyebrow">working notes</span>
                <textarea
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  maxLength={20_000}
                  rows={4}
                  placeholder="What decision or research direction does this project support?"
                  className="resize-y border border-[var(--rule)] bg-transparent p-3 text-[0.875rem] leading-relaxed focus:border-[var(--ink)] focus:outline-none"
                />
              </label>
              <label className="flex flex-col gap-2">
                <span className="eyebrow">tags</span>
                <input
                  value={tags}
                  onChange={(event) => setTags(event.target.value)}
                  placeholder="graphs, fraud detection, review"
                  className="border-b border-[var(--rule)] bg-transparent py-2 text-[0.875rem] focus:border-[var(--ink)] focus:outline-none"
                />
              </label>
              <div className="flex items-end justify-between gap-3">
                <button
                  type="submit"
                  disabled={busy || !name.trim()}
                  className="eyebrow border border-[var(--ink)] px-3 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:opacity-40"
                >
                  {busy ? "saving…" : "save project"}
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmingDelete(true)}
                  disabled={busy}
                  className="eyebrow text-[var(--ink-faint)] hover:text-[var(--live)] disabled:opacity-40"
                >
                  delete project
                </button>
              </div>
            </form>

            <section className="mt-12">
              <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[var(--rule)] pb-3">
                <div>
                  <span className="eyebrow">project evidence</span>
                  <h2 className="display mt-2 text-[1.35rem]">
                    Attached assessments
                  </h2>
                </div>
                <span className="readout text-[0.75rem] text-[var(--ink-faint)]">
                  {project.assessments.length} current input
                  {project.assessments.length === 1 ? "" : "s"}
                </span>
              </div>

              <form
                onSubmit={addAssessment}
                className="mt-5 flex flex-wrap items-end gap-3"
              >
                <label className="flex min-w-[20rem] flex-1 flex-col gap-1">
                  <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
                    add existing assessment
                  </span>
                  <select
                    value={selectedInputId}
                    onChange={(event) => setSelectedInputId(event.target.value)}
                    disabled={busy}
                    className="border-b border-[var(--ink)] bg-transparent py-2 text-[0.875rem] focus:outline-none disabled:opacity-50"
                  >
                    <option value="">choose a research input</option>
                    {available.map((assessment) => (
                      <option
                        key={assessment.research_input_id}
                        value={assessment.research_input_id}
                      >
                        {assessment.input_preview}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="submit"
                  disabled={!selectedInputId || busy}
                  className="eyebrow border border-[var(--ink)] px-3 py-2 hover:bg-[var(--ink)] hover:text-[var(--panel)] disabled:opacity-40"
                >
                  add to project
                </button>
              </form>

              {project.assessments.length === 0 ? (
                <p className="py-10 text-[0.9375rem] text-[var(--ink-soft)]">
                  No assessments attached yet. Add an existing research input
                  above.
                </p>
              ) : (
                <ul className="mt-6">
                  {project.assessments.map((assessment, index) => (
                    <li
                      key={assessment.research_input_id}
                      className="resolve flex items-start gap-4 border-t border-[var(--rule-soft)] py-6 first:border-t-0"
                      style={{
                        animationDelay: `${Math.min(index * 28, 280)}ms`,
                      }}
                    >
                      <Link
                        href={`/assessments/${assessment.id}`}
                        className="min-w-0 flex-1 hover:opacity-80"
                      >
                        <div className="flex flex-wrap items-baseline justify-between gap-3">
                          <span className="eyebrow">
                            {assessment.input_type === "document"
                              ? "uploaded document"
                              : "research idea"}
                          </span>
                          <span className="readout text-[0.6875rem] text-[var(--ink-faint)]">
                            {new Date(assessment.created_at).toLocaleString()}
                          </span>
                        </div>
                        <p className="mt-2 max-w-[68ch] text-[0.9375rem] leading-relaxed text-[var(--ink)]">
                          {assessment.input_preview}
                        </p>
                        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[0.8125rem] text-[var(--ink-soft)]">
                          <span>
                            {assessment.recommendation ?? "Not assessed"}
                          </span>
                          <span>
                            novelty:{" "}
                            {assessment.novelty_level.replace("_", " ")}
                          </span>
                          <span>
                            feasibility:{" "}
                            {assessment.technical_feasibility_level.replace(
                              "_",
                              " ",
                            )}
                          </span>
                        </div>
                      </Link>
                      <button
                        type="button"
                        onClick={() =>
                          void removeAssessment(assessment.research_input_id)
                        }
                        disabled={busy}
                        className="eyebrow shrink-0 text-[0.6875rem] text-[var(--ink-faint)] hover:text-[var(--live)] disabled:opacity-40"
                      >
                        remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </>
        )}
      </div>
      <ConfirmDialog
        open={confirmingDelete}
        title="Delete this project?"
        description="Its assessments will remain available and can be re-attached to another project."
        confirmLabel="delete project"
        busy={busy}
        onConfirm={deleteProject}
        onCancel={() => setConfirmingDelete(false)}
      />
    </main>
  );
}
