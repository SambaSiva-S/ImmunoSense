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

  // Home location (for the Environment agent)
  const [homeLabel, setHomeLabel] = useState<string | null>(null);
  const [locInput, setLocInput] = useState("");
  const [locSaving, setLocSaving] = useState(false);
  const [locMsg, setLocMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .me()
      .then((m) => {
        setDisease(m.disease);
        setConsents(m.consents || {});
        setHomeLabel(m.home_label ?? null);
      })
      .catch((e) => setError(e instanceof ApiError ? `(${e.status}) ${e.message}` : "Couldn't reach the server."))
      .finally(() => setLoading(false));
  }, []);

  async function saveLocation() {
    const q = locInput.trim();
    if (!q) return;
    setLocSaving(true);
    setLocMsg(null);
    try {
      const r = await api.setProfile({ home_query: q });
      if (r.geocode_error) {
        setLocMsg(r.geocode_error);
      } else if (r.home_label) {
        setHomeLabel(r.home_label);
        setLocInput("");
        setLocMsg(`Saved: ${r.home_label}`);
      }
    } catch {
      setLocMsg("Couldn't save that location.");
    } finally {
      setLocSaving(false);
    }
  }

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
            <div className="step-label" style={{ fontSize: 18 }}>Location</div>
            <div style={{ fontSize: 12.5, color: "var(--ink-soft)", margin: "4px 0 12px", lineHeight: 1.45 }}>
              Your home city or zip lets ImmunoSense factor in local air quality and pollen.
            </div>
            <Field label="Current" value={homeLabel ?? "Not set"} />
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <input
                value={locInput}
                onChange={(e) => setLocInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") saveLocation(); }}
                placeholder="City or zip (e.g. Charlotte or 28202)"
                style={{
                  flex: 1, padding: "11px 13px", borderRadius: 12, fontSize: 14,
                  border: "1px solid var(--line)", background: "var(--bg)", color: "var(--ink)",
                }}
              />
              <button
                onClick={saveLocation}
                disabled={locSaving || !locInput.trim()}
                style={{
                  padding: "11px 16px", borderRadius: 12, border: "none", fontSize: 14, fontWeight: 600,
                  background: "var(--sage-deep)", color: "#fff", cursor: "pointer",
                  opacity: locSaving || !locInput.trim() ? 0.6 : 1,
                }}
              >
                {locSaving ? "…" : "Save"}
              </button>
            </div>
            {locMsg && (
              <div style={{ fontSize: 12.5, marginTop: 8,
                            color: locMsg.startsWith("Saved") ? "var(--sage-deep)" : "var(--warn, #b4694a)" }}>
                {locMsg}
              </div>
            )}
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
