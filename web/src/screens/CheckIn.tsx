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
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [photoName, setPhotoName] = useState<string | null>(null);
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);
  const [photoBusy, setPhotoBusy] = useState(false);
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

  // Compress/resize a photo client-side before upload: cap the longest edge and
  // re-encode as JPEG. Keeps it crisp for the user's own record while turning a
  // multi-MB phone photo into a few hundred KB. Falls back to the original file
  // if anything goes wrong (e.g. an unusual format).
  async function compressImage(file: File, maxEdge = 1600, quality = 0.82): Promise<Blob> {
    try {
      const bitmap = await createImageBitmap(file);
      const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
      const w = Math.round(bitmap.width * scale);
      const h = Math.round(bitmap.height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return file;
      ctx.drawImage(bitmap, 0, 0, w, h);
      const blob: Blob | null = await new Promise((res) =>
        canvas.toBlob(res, "image/jpeg", quality)
      );
      // Only use the compressed version if it actually came out smaller.
      return blob && blob.size < file.size ? blob : file;
    } catch {
      return file; // unsupported format etc. — upload the original
    }
  }

  async function handlePhoto(file: File) {
    setError(null);
    setPhotoBusy(true);
    try {
      const compressed = await compressImage(file);
      // 1. ask the API for a signed upload URL + photo_id (always image/jpeg now)
      const { photo_id, upload_url } = await api.photoUploadUrl("image/jpeg");
      // 2. upload the bytes straight to storage (not through our API).
      //    In dev the URL is a stub (dev.local) — skip the actual PUT there.
      if (!upload_url.startsWith("https://dev.local/")) {
        const put = await fetch(upload_url, { method: "PUT", body: compressed,
          headers: { "Content-Type": "image/jpeg" } });
        if (!put.ok) throw new Error(`upload failed (${put.status})`);
      }
      setPhotoId(photo_id);
      setPhotoName(file.name);
      setPhotoPreview(URL.createObjectURL(compressed));
    } catch (e) {
      setError(e instanceof ApiError ? `(${e.status}) ${e.message}`
        : "Couldn't attach the photo. You can still log the meal as text.");
    } finally {
      setPhotoBusy(false);
    }
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
      if (meal.trim() || photoId) {
        await api.logMeal({ source: "text", description: meal.trim() || "(photo only)", photo_id: photoId });
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
            Describe what you ate — this is what informs the dietary signal. A photo is optional and
            just for your own record (it isn't analyzed).
          </div>
          <textarea
            placeholder="Grilled salmon, brown rice, steamed broccoli…"
            value={meal}
            onChange={(e) => setMeal(e.target.value)}
            style={{ height: 64, marginBottom: 12 }}
          />
          {photoId ? (
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13,
                          color: "var(--sage-deep)", background: "var(--sage-wash)",
                          borderRadius: 12, padding: "10px 13px" }}>
              {photoPreview && (
                <img src={photoPreview} alt="meal"
                     style={{ width: 44, height: 44, borderRadius: 8, objectFit: "cover" }} />
              )}
              <span>📎 {photoName ?? "photo attached"}</span>
              <button className="muted-link" style={{ marginLeft: "auto", fontSize: 12 }}
                onClick={() => { setPhotoId(null); setPhotoName(null); setPhotoPreview(null); }}>remove</button>
            </div>
          ) : (
            <label className="btn-ghost" style={{ border: "1.5px dashed var(--line)", borderRadius: 14,
                     width: "100%", padding: 12, display: "block", textAlign: "center", cursor: "pointer" }}>
              {photoBusy ? "Attaching…" : "📷 Attach a photo (optional)"}
              <input
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                disabled={photoBusy}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handlePhoto(f); }}
              />
            </label>
          )}
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
