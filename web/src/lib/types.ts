// Mirrors the backend response shapes (server/api/schemas.py).
// Keeping these in one place means the mobile app can reuse them later.

export interface DisplayBlock {
  headline: string;
  show_number: boolean;
  band_label: string | null;
}

export interface PatternOut {
  name: string;
  label: string;
  description: string;
}

export interface ReportOut {
  bucket_id: string;
  evaluated_at: string | null;
  confidence_level: string; // "insufficient" | "low" | "moderate" | "high"
  status: string; // "ok" | "insufficient"
  flare_probability: number | null;
  severity_composite: number | null;
  severity_band: string | null;
  matched_patterns: PatternOut[];
  explanation: string | null;
  display: DisplayBlock;
  trace_id: string;
}

export interface LogAck {
  ok: boolean;
  log_id: string | null;
  bucket_id: string;
  trace_id: string;
}

export interface SymptomLogIn {
  source?: string;
  fatigue?: number | null;
  joint_pain?: number | null;
  brain_fog_severity?: number | null;
  gi_distress?: number | null;
  sleep_severity?: number | null;
  energy_severity?: number | null;
  wellness_severity?: number | null;
  free_text?: string | null;
}

export interface MealLogIn {
  source?: string;
  description: string;
  photo_id?: string | null;
}

// Dev inspector — mirrors EvaluationService.debug_view (dev-only).
export interface AgentQualityView {
  raw_confidence: number;
  freshness: number;
  quality: number;
  reported: boolean;
  ok: boolean;
}
export interface AgentView {
  agent_id: string;
  ok: boolean | null;
  error: string | null;
  latency_ms: number | null;
  vector_dim: number | null;
  confidence: number | null;
  alerts: string[];
  quality: AgentQualityView | null;
}
export interface DebugView {
  user_id: string;
  bucket_id: string;
  evaluated_at: string | null;
  trace_id: string;
  reporting_agents: string[];
  agents: AgentView[];
  confidence_level: string;
  overall_quality: number;
  flare_probability: number | null;
  severity_composite: number | null;
  severity_band: string | null;
  matched_patterns: PatternOut[];
  fusion_contributions: unknown[];
  embedding_concat_dim: number | null;
  calibration_version: string | null;
  tfm_ok: boolean | null;
  explanation: string | null;
  warnings: string[];
  errors: string[];
}

// UI-friendly confidence pill mapping (presentation only; the backend remains
// the source of truth for show_number).
export function confidenceLabel(level: string): { text: string; tone: "good" | "build" } {
  switch (level) {
    case "high":
      return { text: "Confidence: High", tone: "good" };
    case "moderate":
      return { text: "Confidence: Good", tone: "good" };
    case "low":
      return { text: "Confidence: Building", tone: "build" };
    default:
      return { text: "Confidence: Building", tone: "build" };
  }
}
