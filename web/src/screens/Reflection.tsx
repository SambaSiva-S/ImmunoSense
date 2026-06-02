import { confidenceLabel, type ReportOut } from "../lib/types";

export function Reflection({ report, onBack }: { report: ReportOut; onBack: () => void }) {
  const pill = confidenceLabel(report.confidence_level);
  const insufficient = report.status === "insufficient" || !report.display.show_number;

  return (
    <div className="phone">
      <div className="eyebrow-row">
        <span className="eyebrow">Your reflection · this week</span>
        <span className={`conf-pill ${pill.tone === "good" ? "conf-good" : "conf-build"}`}>
          <span className="pdot" />
          {pill.text}
        </span>
      </div>

      <div className="headline">{report.display.headline}</div>

      {insufficient ? (
        <div className="insufficient">
          <div className="ic">🌱</div>
          <p>
            A few more check-ins over the coming days will let ImmunoSense notice your personal
            rhythms. There's nothing to read into yet — just keep logging when it suits you.
          </p>
        </div>
      ) : (
        <>
          {report.display.show_number && report.flare_probability != null && (
            <div className="num-row">
              <span className="num">
                Estimated likelihood this period: <b>{report.severity_band ?? "—"}</b>
              </span>
            </div>
          )}

          {report.explanation && (
            <div className="why">
              <h3>What's behind this</h3>
              <p>{report.explanation}</p>
            </div>
          )}

          {report.matched_patterns.length > 0 && (
            <div className="patterns">
              {report.matched_patterns.map((p) => (
                <div className="pat" key={p.name}>
                  <div className="pat-ic">🍃</div>
                  <div>
                    <b>{p.label}</b>
                    <span>{p.description}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {insufficient && report.explanation && (
        <div className="why">
          <h3>Why no number?</h3>
          <p>{report.explanation}</p>
        </div>
      )}

      <div className="actions" style={{ marginTop: 18 }}>
        <button className="btn-primary" onClick={onBack}>Back to check-in</button>
      </div>

      <div className="disclaimer">
        ImmunoSense is a wellness companion, not a medical device. It doesn't diagnose or treat
        conditions. For health concerns, talk with your clinician.
      </div>
    </div>
  );
}
