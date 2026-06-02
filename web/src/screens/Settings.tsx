import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import { supabase } from "../lib/supabase";

const CONSENTS: { key: string; label: string; help: string }[] = [
  {
    key: "tfm_ai_processing",
    label: "AI-written explanations",
    help: "Let an AI compose the plain-language 'what's behind this' notes. Turn off to keep explanations strictly rule-based.",
  },
  {
    key: "research_secondary_use",
    label: "Contribute to research",
    help: "Allow your de-identified data to support autoimmune research. Off by default; entirely optional.",
  },
];

export function Settings({ onBack, email }: { onBack: () => void; email: string }) {
  const [disease, setDisease] = useState<string | null>(null);
  const [consents, setConsents] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    api
      .me()
      .then((m) => {
        setDisease(m.disease);
        setConsents(m.consents || {});
      })
      .catch((e) => setError(e instanceof ApiError ? `(${e.status}) ${e.message}` : "Couldn't reach the server."))
      .finally(() => setLoading(false));
  }, []);

  async function toggle(key: string) {
    const next = !consents[key];
    setConsents((c) => ({ ...c, [key]: next })); // optimistic
    setSaving(key);
    try {
      await api.setConsent(key, next);
    } catch {
      setConsents((c) => ({ ...c, [key]: !next })); // revert on failure
      setError("Couldn't save that change.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className="phone">
      <div className="top">
        <span className="eyebrow">Settings</span>
        <button className="muted-link" onClick={onBack}>← back</button>
      </div>

      {error && <div className="err" style={{ marginTop: 14 }}>{error}</div>}

      {loading ? (
        <div className="loading"><div className="spinner" />Loading…</div>
      ) : (
        <>
          <div style={{ marginTop: 16 }}>
            <div className="step-label" style={{ fontSize: 18 }}>Profile</div>
            <Field label="Signed in as" value={email} />
            <Field label="Condition" value={disease ?? "Not set"} />
          </div>

          <div style={{ marginTop: 22 }}>
            <div className="step-label" style={{ fontSize: 18 }}>Privacy &amp; AI</div>
            {CONSENTS.map((c) => (
              <div key={c.key} style={{ padding: "14px 0", borderBottom: "1px solid var(--line)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
                  <div style={{ fontSize: 14.5, fontWeight: 500 }}>{c.label}</div>
                  <Toggle on={!!consents[c.key]} busy={saving === c.key} onClick={() => toggle(c.key)} />
                </div>
                <div style={{ fontSize: 12.5, color: "var(--ink-soft)", marginTop: 5, lineHeight: 1.45 }}>{c.help}</div>
              </div>
            ))}
          </div>

          <div className="actions" style={{ marginTop: 24 }}>
            <button className="btn-primary" onClick={onBack}>Done</button>
          </div>
          <button className="skip-today" onClick={() => supabase.auth.signOut()}>Sign out</button>

          <div className="disclaimer">
            You can change these anytime. Turning off AI explanations keeps your reflections strictly
            rule-based; your numbers and trends are unaffected.
          </div>
        </>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--line)" }}>
      <span style={{ color: "var(--ink-soft)", fontSize: 14 }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function Toggle({ on, busy, onClick }: { on: boolean; busy: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={busy}
      aria-pressed={on}
      style={{
        width: 46, height: 27, borderRadius: 999, border: "none", cursor: "pointer",
        background: on ? "var(--sage-deep)" : "var(--line)", position: "relative", transition: ".2s",
        opacity: busy ? 0.6 : 1, flexShrink: 0,
      }}
    >
      <span
        style={{
          position: "absolute", top: 3, left: on ? 22 : 3, width: 21, height: 21, borderRadius: "50%",
          background: "#fff", transition: ".2s", boxShadow: "0 1px 3px rgba(0,0,0,.2)",
        }}
      />
    </button>
  );
}
