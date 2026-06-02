import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase } from "./lib/supabase";
import { Auth } from "./screens/Auth";
import { CheckIn } from "./screens/CheckIn";
import { Reflection } from "./screens/Reflection";
import { Inspector } from "./screens/Inspector";
import { History } from "./screens/History";
import { Settings } from "./screens/Settings";
import { Onboarding } from "./screens/Onboarding";
import { Biomarker } from "./screens/Biomarker";
import { api } from "./lib/api";
import type { ReportOut } from "./lib/types";

type View = "checkin" | "reflection" | "skipped" | "inspector" | "history" | "settings" | "onboarding" | "biomarker";

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);
  const [view, setView] = useState<View>("checkin");
  const [report, setReport] = useState<ReportOut | null>(null);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setReady(true);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => sub.subscription.unsubscribe();
  }, []);

  // On sign-in, check whether the profile has a condition; if not, onboard.
  useEffect(() => {
    if (!session) return;
    api.me()
      .then((m) => setNeedsOnboarding(!m.disease))
      .catch(() => setNeedsOnboarding(false)); // if /me fails, don't block the app
  }, [session]);

  if (!ready) {
    return (
      <div className="app">
        <div className="phone loading">
          <div className="spinner" />
          Warming up…
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="app">
        <Auth />
      </div>
    );
  }

  const name = session.user.email?.split("@")[0] ?? "there";

  if (needsOnboarding) {
    return (
      <div className="app">
        <Onboarding onDone={() => { setNeedsOnboarding(false); setView("checkin"); }} />
      </div>
    );
  }

  return (
    <div className="app">
      {view === "checkin" && (
        <CheckIn
          greetingName={name}
          onReflection={(r) => { setReport(r); setView("reflection"); }}
          onSkip={() => setView("skipped")}
        />
      )}
      {view === "reflection" && report && (
        <Reflection report={report} onBack={() => setView("checkin")} />
      )}
      {view === "inspector" && <Inspector onBack={() => setView("checkin")} />}
      {view === "history" && <History onBack={() => setView("checkin")} />}
      {view === "settings" && <Settings onBack={() => setView("checkin")} email={session.user.email ?? ""} />}
      {view === "biomarker" && <Biomarker onBack={() => setView("checkin")} />}
      {view === "skipped" && (
        <div className="phone center-card">
          <div className="brand-mark">🌙</div>
          <h1>See you soon</h1>
          <p className="sub" style={{ marginBottom: 22 }}>
            Skipped for today — nothing's lost by taking a break. We're here whenever you're ready.
          </p>
          <div className="actions">
            <button className="btn-primary" onClick={() => setView("checkin")}>Back to check-in</button>
          </div>
        </div>
      )}

      {/* Simple bottom nav across the main sections */}
      {view !== "inspector" && (
        <nav style={{ display: "flex", justifyContent: "center", gap: 6, marginTop: 16 }}>
          <NavBtn active={view === "checkin" || view === "reflection" || view === "skipped"} onClick={() => setView("checkin")}>Today</NavBtn>
          <NavBtn active={view === "history"} onClick={() => setView("history")}>History</NavBtn>
          <NavBtn active={view === "settings"} onClick={() => setView("settings")}>Settings</NavBtn>
        </nav>
      )}

      {/* Quick links from the main views */}
      {(view === "checkin" || view === "reflection") && (
        <div style={{ textAlign: "center", marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
          <button className="muted-link" style={{ fontSize: 13 }} onClick={() => setView("biomarker")}>
            ＋ Add a lab result
          </button>
          <button className="muted-link" style={{ fontSize: 12, opacity: 0.6 }} onClick={() => setView("inspector")}>
            ⚙ dev: inspect agents
          </button>
        </div>
      )}
    </div>
  );
}

function NavBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "inherit", fontSize: 13, fontWeight: 500, border: "1px solid var(--line)",
        background: active ? "var(--sage)" : "var(--surface)", color: active ? "#fff" : "var(--ink-soft)",
        padding: "9px 20px", borderRadius: 999, cursor: "pointer", transition: ".2s",
      }}
    >
      {children}
    </button>
  );
}
