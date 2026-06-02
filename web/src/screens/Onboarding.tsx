import { useState } from "react";
import { api, ApiError } from "../lib/api";

const CONDITIONS = [
  { code: "SLE", label: "Lupus (SLE)" },
  { code: "RA", label: "Rheumatoid arthritis" },
  { code: "PSA", label: "Psoriatic arthritis" },
  { code: "IBD", label: "IBD / Crohn's / Colitis" },
  { code: "MS", label: "Multiple sclerosis" },
  { code: "OTHER", label: "Another condition" },
];

export function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(1);
  const [condition, setCondition] = useState<string | null>(null);
  const [sex, setSex] = useState<number | null>(null);
  const [dob, setDob] = useState("");
  const [heightUnit, setHeightUnit] = useState<"cm" | "ftin">("cm");
  const [heightCm, setHeightCm] = useState("");
  const [heightFt, setHeightFt] = useState("");
  const [heightIn, setHeightIn] = useState("");
  const [weightUnit, setWeightUnit] = useState<"kg" | "lb">("kg");
  const [weightVal, setWeightVal] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const guessedTz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

  function computeHeightCm(): number | null {
    if (heightUnit === "cm") {
      const v = parseFloat(heightCm);
      return isFinite(v) && v > 0 ? v : null;
    }
    const ft = parseFloat(heightFt) || 0;
    const inch = parseFloat(heightIn) || 0;
    const cm = (ft * 12 + inch) * 2.54;
    return cm > 0 ? Math.round(cm * 10) / 10 : null;
  }
  function computeWeightKg(): number | null {
    const v = parseFloat(weightVal);
    if (!isFinite(v) || v <= 0) return null;
    return weightUnit === "kg" ? v : Math.round(v * 0.453592 * 10) / 10;
  }

  async function finish() {
    setError(null);
    setBusy(true);
    try {
      await api.setProfile({
        disease: condition,
        timezone: guessedTz,
        sex,
        date_of_birth: dob || null,
        height_cm: computeHeightCm(),
        weight_kg: computeWeightKg(),
      });
      onDone();
    } catch (e) {
      setError(e instanceof ApiError ? `(${e.status}) ${e.message}` : "Couldn't reach the server.");
      setBusy(false);
    }
  }

  const hCm = computeHeightCm();
  const wKg = computeWeightKg();
  const bmiPreview = hCm && wKg ? (wKg / Math.pow(hCm / 100, 2)).toFixed(1) : null;

  return (
    <div className="phone">
      <div className="brand-mark" style={{ textAlign: "center" }}>🌿</div>
      <h1 style={{ textAlign: "center" }}>Welcome to<br />ImmunoSense</h1>
      <p className="sub" style={{ textAlign: "center", marginBottom: 8 }}>
        A calm companion for noticing how you feel over time.
      </p>

      <div className="dots" style={{ marginTop: 18 }}>
        <span className={step >= 1 ? "now" : ""} /><span className={step >= 2 ? "now" : ""} />
      </div>

      {error && <div className="err">{error}</div>}

      {step === 1 && (
        <>
          <div className="step-label">Which condition are you tracking?</div>
          <div className="step-hint">This tailors your reflections. You can change it later.</div>
          <div className="chips" style={{ marginBottom: 22 }}>
            {CONDITIONS.map((c) => (
              <button key={c.code} className={condition === c.code ? "sel" : ""} onClick={() => setCondition(c.code)}>
                {c.label}
              </button>
            ))}
          </div>
          <div className="actions">
            <button className="btn-ghost" onClick={onDone} disabled={busy}>Skip setup</button>
            <button className="btn-primary" onClick={() => setStep(2)} disabled={!condition}>Next &rarr;</button>
          </div>
        </>
      )}

      {step === 2 && (
        <>
          <div className="step-label">A little about you</div>
          <div className="step-hint">
            These help interpret your lab values accurately &mdash; inflammation markers read differently
            by age, sex, and body size. Optional, but they make your reflections more personal.
          </div>

          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 7 }}>Sex (for lab baselines)</div>
          <div className="chips" style={{ marginBottom: 18 }}>
            <button className={sex === 1 ? "sel" : ""} onClick={() => setSex(1)}>Male</button>
            <button className={sex === 2 ? "sel" : ""} onClick={() => setSex(2)}>Female</button>
          </div>

          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 7 }}>Date of birth</div>
          <input className="field" type="date" value={dob} onChange={(e) => setDob(e.target.value)} />

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, marginBottom: 7 }}>
            <span style={{ fontSize: 14, fontWeight: 500 }}>Height</span>
            <UnitToggle a="cm" b="ft / in" active={heightUnit === "cm" ? "a" : "b"}
              onPick={(s) => setHeightUnit(s === "a" ? "cm" : "ftin")} />
          </div>
          {heightUnit === "cm" ? (
            <input className="field" type="number" inputMode="decimal" placeholder="cm"
              value={heightCm} onChange={(e) => setHeightCm(e.target.value)} />
          ) : (
            <div style={{ display: "flex", gap: 10 }}>
              <input className="field" type="number" inputMode="numeric" placeholder="ft"
                value={heightFt} onChange={(e) => setHeightFt(e.target.value)} style={{ flex: 1 }} />
              <input className="field" type="number" inputMode="numeric" placeholder="in"
                value={heightIn} onChange={(e) => setHeightIn(e.target.value)} style={{ flex: 1 }} />
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, marginBottom: 7 }}>
            <span style={{ fontSize: 14, fontWeight: 500 }}>Weight</span>
            <UnitToggle a="kg" b="lb" active={weightUnit === "kg" ? "a" : "b"}
              onPick={(s) => setWeightUnit(s === "a" ? "kg" : "lb")} />
          </div>
          <input className="field" type="number" inputMode="decimal" placeholder={weightUnit}
            value={weightVal} onChange={(e) => setWeightVal(e.target.value)} />

          {bmiPreview && (
            <div style={{ fontSize: 13, color: "var(--sage-deep)", background: "var(--sage-wash)",
                          borderRadius: 12, padding: "10px 13px", marginBottom: 8 }}>
              BMI: <b>{bmiPreview}</b> &mdash; used only to interpret your markers, never shown as a judgment.
            </div>
          )}

          <div style={{ fontSize: 12, color: "var(--ink-soft)", margin: "12px 0 18px", lineHeight: 1.5,
                        borderLeft: "2px solid var(--line)", paddingLeft: 12 }}>
            Timezone detected: {guessedTz}. ImmunoSense is a wellness companion &mdash; it doesn't diagnose
            or treat. For medical concerns, your clinician is the right place.
          </div>

          <div className="actions">
            <button className="btn-ghost" onClick={() => setStep(1)} disabled={busy}>&larr; Back</button>
            <button className="btn-primary" onClick={finish} disabled={busy}>
              {busy ? "Setting up…" : "Get started \u2192"}
            </button>
          </div>
          <button className="skip-today" onClick={finish} disabled={busy}>Skip the details for now</button>
        </>
      )}
    </div>
  );
}

function UnitToggle({ a, b, active, onPick }: {
  a: string; b: string; active: "a" | "b"; onPick: (s: "a" | "b") => void;
}) {
  return (
    <div style={{ display: "flex", gap: 4, background: "var(--bg)", borderRadius: 999, padding: 3 }}>
      {(["a", "b"] as const).map((s) => (
        <button key={s} onClick={() => onPick(s)}
          style={{
            fontFamily: "inherit", fontSize: 12, fontWeight: 500, border: "none", cursor: "pointer",
            padding: "5px 12px", borderRadius: 999,
            background: active === s ? "var(--sage)" : "transparent",
            color: active === s ? "#fff" : "var(--ink-soft)",
          }}>
          {s === "a" ? a : b}
        </button>
      ))}
    </div>
  );
}
