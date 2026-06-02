import { useState } from "react";
import { api, ApiError } from "../lib/api";
import type { ReportOut } from "../lib/types";

const MOODS = ["Calm", "Foggy", "Low", "Anxious", "Okay"];

// The structured symptom fields the agent consumes, as gentle 1-5 taps.
// Higher tap = more severe (matches the backend severity convention).
const SYMPTOMS: { key: string; label: string; lowLabel: string; highLabel: string }[] = [
  { key: "fatigue", label: "Fatigue", lowLabel: "None", highLabel: "Severe" },
  { key: "joint_pain", label: "Joint pain", lowLabel: "None", highLabel: "Severe" },
  { key: "sleep_severity", label: "Sleep trouble", lowLabel: "Slept well", highLabel: "Very poor" },
  { key: "brain_fog_severity", label: "Brain fog", lowLabel: "Clear", highLabel: "Heavy" },
  { key: "gi_distress", label: "Gut / digestion", lowLabel: "Fine", highLabel: "Rough" },
];

export function CheckIn({
  onReflection,
  onSkip,
  greetingName,
}: {
  onReflection: (r: ReportOut) => void;
  onSkip: () => void;
  greetingName: string;
}) {
  // Each symptom is optional; null = not answered. Default the first to a mid value.
  const [symptoms, setSymptoms] = useState<Record<string, number | null>>({
    fatigue: 2,
    joint_pain: null,
    sleep_severity: null,
    brain_fog_severity: null,
    gi_distress: null,
  });
  const [moods, setMoods] = useState<string[]>(["Calm"]);
  const [note, setNote] = useState("");
  const [meal, setMeal] = useState("");
  const [showMeal, setShowMeal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const today = new Date().toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });

  function setSymptom(key: string, val: number) {
    setSymptoms((cur) => ({ ...cur, [key]: cur[key] === val ? null : val }));
  }

  function toggleMood(m: string) {
    setMoods((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]));
  }

  // Map a 1-5 tap to the backend's severity scale (roughly 0-10).
  function tapToSeverity(v: number | null): number | undefined {
    if (v == null) return undefined;
    return (v - 1) * 2.5; // 1->0, 5->10
  }

  async function submit(asFlare = false) {
    setError(null);
    setBusy(true);
    try {
      const moodNote = moods.length ? `Mood: ${moods.join(", ")}.` : "";
      const freeText = [moodNote, note].filter(Boolean).join(" ") || undefined;
      await api.logSymptom({
        source: "tap",
        fatigue: tapToSeverity(symptoms.fatigue),
        joint_pain: tapToSeverity(symptoms.joint_pain),
        sleep_severity: tapToSeverity(symptoms.sleep_severity),
        brain_fog_severity: tapToSeverity(symptoms.brain_fog_severity),
        gi_distress: tapToSeverity(symptoms.gi_distress),
        free_text: freeText,
      });
      if (meal.trim()) {
        await api.logMeal({ source: "text", description: meal.trim() });
      }
      const report = asFlare ? await api.logFlare(0.8) : await api.evaluate();
      onReflection(report);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 401
            ? "Your session needs a refresh — please sign in again."
            : `Couldn't save that (${e.status}): ${e.message}`
        );
      } else {
        setError("Couldn't reach the server. Is the API running on the configured URL?");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="phone">
      <div className="top">
        <span className="eyebrow">Hello, {greetingName}</span>
        <span className="date">{today}</span>
      </div>
      <h1>How are you<br />feeling today?</h1>
      <p className="sub">A quick check-in — answer what feels relevant, skip the rest.</p>

      {error && <div className="err" style={{ marginTop: 18 }}>{error}</div>}

      <div className="dots">
        <span className="done" /><span className="now" /><span /><span />
      </div>

      <div className="step-label">How's your body today?</div>
      <div className="step-hint">Tap a level for whatever's relevant — skip the rest.</div>

      {SYMPTOMS.map((s) => (
        <div key={s.key} style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 7 }}>{s.label}</div>
          <div className="scale">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                className={symptoms[s.key] === n ? "sel" : ""}
                onClick={() => setSymptom(s.key, n)}
              >
                {n}
              </button>
            ))}
          </div>
          <div className="scale-ends" style={{ marginBottom: 0 }}>
            <span>{s.lowLabel}</span><span>{s.highLabel}</span>
          </div>
        </div>
      ))}

      <div style={{ height: 8 }} />

      <div className="step-label">How's your mood?</div>
      <div className="step-hint">Tap any that fit.</div>
      <div className="chips">
        {MOODS.map((m) => (
          <button key={m} className={moods.includes(m) ? "sel" : ""} onClick={() => toggleMood(m)}>
            {m}
          </button>
        ))}
      </div>

      <div className="step-label" style={{ fontSize: 20 }}>Anything else on your mind?</div>
      <div className="step-hint">Optional — type a note about your day.</div>
      <textarea
        placeholder="Slept poorly, knees a little stiff this morning…"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />

      {!showMeal ? (
        <button
          className="btn-ghost"
          style={{ border: "1.5px dashed var(--line)", borderRadius: 14, width: "100%", padding: 13, marginBottom: 24 }}
          onClick={() => setShowMeal(true)}
        >
          ＋ Add a meal
        </button>
      ) : (
        <div style={{ marginBottom: 24 }}>
          <div className="step-hint" style={{ marginBottom: 8 }}>
            Describe what you ate — this is what informs the dietary signal. A photo (coming soon)
            is just for your own record.
          </div>
          <textarea
            placeholder="Grilled salmon, brown rice, steamed broccoli…"
            value={meal}
            onChange={(e) => setMeal(e.target.value)}
            style={{ height: 64, marginBottom: 0 }}
          />
        </div>
      )}

      <div className="actions">
        <button className="btn-ghost" onClick={onSkip} disabled={busy}>Skip</button>
        <button className="btn-primary" onClick={() => submit(false)} disabled={busy}>
          {busy ? "Reflecting…" : "Continue →"}
        </button>
      </div>

      <button className="skip-today" onClick={onSkip} disabled={busy}>Skip for today →</button>

      <div className="flare" onClick={() => !busy && submit(true)}>
        <div className="flare-dot" />
        <div className="flare-txt">
          <b>Having a flare right now?</b>Tap to log it and get a reflection straight away.
        </div>
      </div>
    </div>
  );
}
