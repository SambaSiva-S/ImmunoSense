import { useState } from "react";
import { api, ApiError } from "../lib/api";
import type { DebugView } from "../lib/types";

export function Inspector({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<DebugView | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setError(null);
    setBusy(true);
    try {
      setData(await api.evaluateDebug());
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 404
            ? "Debug endpoint disabled. Start the API with ENABLE_DEBUG_ENDPOINT=1."
            : `(${e.status}) ${e.message}`
        );
      } else {
        setError("Couldn't reach the server.");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="phone" style={{ fontFamily: "monospace" }}>
      <div className="top">
        <span className="eyebrow" style={{ color: "var(--clay)" }}>⚙ Dev · Agent inspector</span>
        <button className="muted-link" onClick={onBack}>← back</button>
      </div>
      <p className="sub" style={{ marginTop: 8, marginBottom: 16 }}>
        Builder-only view of what the Conductor computed — the internals the wellness screen hides.
        Not visible to real users.
      </p>

      <button className="btn-primary" onClick={run} disabled={busy} style={{ width: "100%" }}>
        {busy ? "Evaluating…" : "Run evaluation & inspect"}
      </button>

      {error && <div className="err" style={{ marginTop: 14 }}>{error}</div>}

      {data && (
        <div style={{ marginTop: 18, fontSize: 12.5, lineHeight: 1.7 }}>
          <Row k="bucket" v={data.bucket_id} />
          <Row k="confidence" v={data.confidence_level} />
          <Row k="overall_quality" v={data.overall_quality.toFixed(3)} />
          <Row k="flare_probability" v={String(data.flare_probability)} />
          <Row k="severity_composite" v={String(data.severity_composite)} />
          <Row k="severity_band" v={String(data.severity_band)} />
          <Row k="calibration" v={String(data.calibration_version)} />
          <Row k="tfm_ok" v={String(data.tfm_ok)} />
          <Row k="embedding_dim" v={String(data.embedding_concat_dim)} />

          <div style={{ marginTop: 14, fontWeight: 700, color: "var(--sage-deep)" }}>
            AGENTS ({data.reporting_agents.length})
          </div>
          {data.agents.map((a) => (
            <div key={a.agent_id} style={{ background: "var(--sage-wash)", borderRadius: 10, padding: "9px 11px", margin: "7px 0" }}>
              <div style={{ fontWeight: 700 }}>{a.agent_id}</div>
              <div>ok={String(a.ok)} · dim={String(a.vector_dim)} · conf={String(a.confidence)} · {a.latency_ms ?? "?"}ms</div>
              {a.quality && (
                <div style={{ color: "var(--ink-soft)" }}>
                  quality={a.quality.quality.toFixed(3)} · fresh={a.quality.freshness.toFixed(2)} · reported={String(a.quality.reported)}
                </div>
              )}
              {a.alerts.length > 0 && <div style={{ color: "var(--clay)" }}>alerts: {a.alerts.join(", ")}</div>}
              {a.error && <div style={{ color: "var(--clay)" }}>error: {a.error}</div>}
            </div>
          ))}

          <div style={{ marginTop: 12, fontWeight: 700, color: "var(--sage-deep)" }}>
            PATTERNS ({data.matched_patterns.length})
          </div>
          {data.matched_patterns.length === 0
            ? <div style={{ color: "var(--ink-soft)" }}>none matched</div>
            : data.matched_patterns.map((p) => <div key={p.name}>· {p.name} — {p.label}</div>)}

          {data.warnings.length > 0 && (
            <>
              <div style={{ marginTop: 12, fontWeight: 700, color: "var(--clay)" }}>WARNINGS</div>
              {data.warnings.map((w, i) => <div key={i} style={{ color: "var(--clay)" }}>· {w}</div>)}
            </>
          )}

          <div style={{ marginTop: 14, color: "var(--ink-soft)" }}>trace: {data.trace_id}</div>
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--line)", padding: "3px 0" }}>
      <span style={{ color: "var(--ink-soft)" }}>{k}</span>
      <span style={{ fontWeight: 600 }}>{v}</span>
    </div>
  );
}
