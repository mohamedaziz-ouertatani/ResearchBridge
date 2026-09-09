import { API_BASE } from "./api";
import type { AssessmentSummary } from "./assessmentApi";

export type ResearchProjectSummary = {
  id: string;
  name: string;
  notes: string | null;
  tags: string[];
  created_at: string;
  updated_at: string;
  assessment_count: number;
};

export type ResearchProject = {
  id: string;
  name: string;
  notes: string | null;
  tags: string[];
  created_at: string;
  updated_at: string;
  assessments: AssessmentSummary[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

async function requestNoContent(
  path: string,
  init?: RequestInit,
): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
}

export const projectApi = {
  list: () => request<ResearchProjectSummary[]>("/api/projects"),
  create: (name: string, notes: string | null = null, tags: string[] = []) =>
    request<ResearchProjectSummary>("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, notes, tags }),
    }),
  get: (id: string) => request<ResearchProject>(`/api/projects/${id}`),
  update: (
    id: string,
    payload: { name?: string; notes?: string | null; tags?: string[] },
  ) =>
    request<ResearchProjectSummary>(`/api/projects/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  remove: (id: string) =>
    requestNoContent(`/api/projects/${id}`, { method: "DELETE" }),
  addAssessment: (projectId: string, researchInputId: string) =>
    requestNoContent(
      `/api/projects/${projectId}/assessments/${researchInputId}`,
      { method: "PUT" },
    ),
  removeAssessment: (projectId: string, researchInputId: string) =>
    requestNoContent(
      `/api/projects/${projectId}/assessments/${researchInputId}`,
      { method: "DELETE" },
    ),
};
