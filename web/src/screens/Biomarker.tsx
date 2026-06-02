import { useState } from "react";
import { api, ApiError } from "../lib/api";

export function Biomarker({ onBack }: { onBack: () => void }) {
  const [crp, setCrp] = useState("");
  const [esr, setEsr] = useState("");
  const today = new Date().toISOString().slice(0, 10);
  const [measuredAt, setMeasuredAt] = useState(today);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function save() {
    setError(null);
    setBusy(true);
    try {
      const body: { crp?: number | null; esr?: number | null; measured_at?: string | null } = {};
      if (crp.trim()) body.crp = parseFloat(crp);
      if (esr.trim()) body.esr = parseFloat(esr);
      if (body.crp == null && body.esr == null) {
        setError("Enter at least one value.");
        setBusy(false);
        return;
      }
      if (measuredAt && measuredAt !== today) body.measured_at = measuredAt;
      await api.logBiomarker(body);
      setSaved(true);
      setCrp("");
      setEsr("");
    } catch (e) {
      setError(e instanceof ApiError ? `(${e.status}) ${e.message}` : "Couldn't reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="phone">
      <div className="top">
        <span className="eyebrow">Lab values</span>
        <button className="muted-link" onClick={onBack}>← back</button>
      </div>
      <h1 style={{ fontSize: 26 }}>Add a lab result</h1>
      <p className="sub">
        Got recent bloodwork? Enter the numbers from your report. Optional — only what you have.
      </p>

      {saved && (
        <div className="err" style={{ background: "var(--sage-wash)", borderColor: "#e2e6da", color: "var(--sage-deep)", marginTop: 16 }}>
          Saved. These feed into your next reflection.
        </div>
      )}
      {error && <div className="err" style={{ marginTop: 16 }}>{error}</div>}

      <div style={{ marginTop: 20 }}>
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 6 }}>When was this taken?</div>
          <input
            className="field"
            type="date"
            value={measuredAt}
            max={today}
            onChange={(e) => setMeasuredAt(e.target.value)}
            style={{ marginBottom: 4 }}
          />
          <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            Defaults to today. Have older bloodwork? Set the date it was drawn — past results help
            build your baseline faster.
          </div>
        </div>
        <LabField
          label="CRP"
          unit="mg/L"
          hint="C-reactive protein — a common inflammation marker."
          value={crp}
          onChange={setCrp}
        />
        <LabField
          label="ESR"
          unit="mm/hr"
          hint="Erythrocyte sedimentation rate."
          value={esr}
          onChange={setEsr}
        />
      </div>

      <div className="actions" style={{ marginTop: 8 }}>
        <button className="btn-ghost" onClick={onBack} disabled={busy}>Back</button>
        <button className="btn-primary" onClick={save} disabled={busy}>
          {busy ? "Saving…" : "Save lab values"}
        </button>
      </div>

      <div className="disclaimer">
        Enter values exactly as they appear on your lab report. ImmunoSense uses them to enrich your
        wellness picture — it doesn't interpret labs as a clinician would.
      </div>
    </div>
  );
}

function LabField({
  label, unit, hint, value, onChange,
}: {
  label: string; unit: string; hint: string; value: string; onChange: (v: string) => void;
}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontSize: 15, fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>{unit}</span>
      </div>
      <input
        className="field"
        type="number"
        inputMode="decimal"
        placeholder="—"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ marginBottom: 4 }}
      />
      <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>{hint}</div>
    </div>
  );
}
