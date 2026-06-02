import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import { confidenceLabel, type ReportOut } from "../lib/types";

export function History({ onBack }: { onBack: () => void }) {
  const [items, setItems] = useState<ReportOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .history(60)
      .then((r) => setItems(r.items))
      .catch((e) => {
        setError(e instanceof ApiError ? `(${e.status}) ${e.message}` : "Couldn't reach the server.");
      });
  }, []);

  if (error) {
    return (
      <div className="phone">
        <Header onBack={onBack} />
        <div className="err" style={{ marginTop: 16 }}>{error}</div>
      </div>
    );
  }

  if (!items) {
    return (
      <div className="phone">
        <Header onBack={onBack} />
        <div className="loading"><div className="spinner" />Gathering your history…</div>
      </div>
    );
  }

  // Oldest -> newest for the chart; severity_composite is the trend signal.
  const chrono = [...items].reverse();
  const points = chrono
    .map((r) => (r.severity_composite == null ? null : r.severity_composite))
    .filter((v): v is number => v != null);

  return (
    <div className="phone">
      <Header onBack={onBack} />

      {items.length === 0 ? (
        <div className="insufficient" style={{ marginTop: 20 }}>
          <div className="ic">🌱</div>
          <p>Your history is just beginning. Each check-in adds a point here — come back after a
            few days to see your rhythms take shape.</p>
        </div>
      ) : (
        <>
          <div className="trend-card" style={{ marginTop: 18 }}>
            <div className="trend-head">
              <span>Your trend</span>
              <span className="band">{items.length} {items.length === 1 ? "reflection" : "reflections"}</span>
            </div>
            {points.length >= 2 ? (
              <Sparkline values={points} />
            ) : (
              <p style={{ fontSize: 13, color: "var(--ink-soft)", fontFamily: "'Newsreader',serif", fontStyle: "italic" }}>
                One data point so far — a line appears once you have a couple more check-ins.
              </p>
            )}
          </div>

          <div style={{ marginTop: 8 }}>
            {chrono.slice().reverse().map((r, i) => (
              <ReflectionRow key={`${r.bucket_id}-${i}`} report={r} />
            ))}
          </div>
        </>
      )}

      <div className="disclaimer">
        ImmunoSense is a wellness companion, not a medical device. Trends reflect what you've logged,
        not a clinical assessment.
      </div>
    </div>
  );
}

function Header({ onBack }: { onBack: () => void }) {
  return (
    <div className="top">
      <span className="eyebrow">Your history</span>
      <button className="muted-link" onClick={onBack}>← back</button>
    </div>
  );
}

function ReflectionRow({ report }: { report: ReportOut }) {
  const pill = confidenceLabel(report.confidence_level);
  const when = report.evaluated_at
    ? new Date(report.evaluated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : report.bucket_id;
  return (
    <div className="pat" style={{ marginBottom: 9, alignItems: "center" }}>
      <div className="pat-ic">{report.status === "insufficient" ? "🌱" : "🍃"}</div>
      <div style={{ flex: 1 }}>
        <b style={{ fontWeight: 500 }}>{report.display.headline}</b>
        <span>{when}</span>
      </div>
      <span className={`conf-pill ${pill.tone === "good" ? "conf-good" : "conf-build"}`} style={{ fontSize: 10.5 }}>
        <span className="pdot" />{report.confidence_level}
      </span>
    </div>
  );
}

/** Minimal SVG sparkline; no chart lib needed for Phase 1. */
function Sparkline({ values }: { values: number[] }) {
  const w = 300, h = 80, pad = 8;
  const max = Math.max(...values, 0.1);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const step = (w - pad * 2) / Math.max(values.length - 1, 1);
  const pts = values
    .map((v, i) => {
      const x = pad + i * step;
      const y = h - pad - ((v - min) / span) * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const lastX = pad + (values.length - 1) * step;
  const lastY = h - pad - ((values[values.length - 1] - min) / span) * (h - pad * 2);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", height: "auto" }}>
      <polyline fill="none" stroke="#5f7050" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" points={pts} />
      <circle cx={lastX} cy={lastY} r="4.5" fill="#5f7050" />
      <circle cx={lastX} cy={lastY} r="9" fill="#5f7050" opacity="0.15" />
    </svg>
  );
}
