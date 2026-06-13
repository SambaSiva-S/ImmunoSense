import { supabase } from "./supabase";
import type { DebugView, LogAck, MealLogIn, ReportOut, SymptomLogIn } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL as string;

/** Get the current access token from the Supabase session. */
async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(await authHeader()),
    ...(init?.headers as Record<string, string> | undefined),
  };
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  // 204 / empty
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  logSymptom: (body: SymptomLogIn) =>
    request<LogAck>("/v1/log/symptom", { method: "POST", body: JSON.stringify(body) }),

  logMeal: (body: MealLogIn) =>
    request<LogAck>("/v1/log/meal", { method: "POST", body: JSON.stringify(body) }),

  photoUploadUrl: (content_type = "image/jpeg") =>
    request<{ photo_id: string; upload_url: string; storage_key: string }>("/v1/photo", {
      method: "POST",
      body: JSON.stringify({ content_type }),
    }),

  photoViewUrl: (photoId: string) =>
    request<{ view_url: string }>(`/v1/photo/${photoId}`),

  evaluate: () => request<ReportOut>("/v1/evaluate", { method: "POST" }),

  evaluateDebug: () => request<DebugView>("/v1/evaluate/debug", { method: "POST" }),

  logFlare: (severity = 0.8) =>
    request<ReportOut>("/v1/log/flare", { method: "POST", body: JSON.stringify({ severity }) }),

  reportLatest: () => request<ReportOut>("/v1/report/latest"),

  history: (limit = 30) =>
    request<{ items: ReportOut[]; trace_id: string }>(`/v1/history?limit=${limit}`),

  me: () =>
    request<{ user_id: string; disease: string | null; consents: Record<string, boolean> }>("/v1/me"),

  setConsent: (consent_type: string, granted: boolean) =>
    request<{ ok: boolean; consent_type: string; granted: boolean }>("/v1/me/consent", {
      method: "PUT",
      body: JSON.stringify({ consent_type, granted }),
    }),

  setProfile: (p: {
    disease?: string | null;
    timezone?: string | null;
    sex?: number | null;
    date_of_birth?: string | null;
    height_cm?: number | null;
    weight_kg?: number | null;
  }) =>
    request<{ ok: boolean; disease: string | null; timezone: string | null }>("/v1/me/profile", {
      method: "PUT",
      body: JSON.stringify(p),
    }),

  logBiomarker: (body: { crp?: number | null; esr?: number | null; measured_at?: string | null; extra?: Record<string, number> }) =>
    request<LogAck>("/v1/log/biomarker", { method: "POST", body: JSON.stringify(body) }),
};
