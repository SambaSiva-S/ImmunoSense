import { useState } from "react";
import { supabase } from "../lib/supabase";

export function Auth() {
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<string | null>(null);

  async function submit() {
    setError(null);
    setInfo(null);
    setBusy(true);
    try {
      if (mode === "signin") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        // onAuthStateChange in App will swap the screen.
      } else {
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) throw error;
        setInfo("Account created. If email confirmation is on, check your inbox — otherwise you can sign in now.");
        setMode("signin");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="phone center-card">
      <div className="brand-mark">🌿</div>
      <h1>ImmunoSense</h1>
      <p className="sub" style={{ marginBottom: 24 }}>
        Your calm companion for tracking how you feel.
      </p>

      {error && <div className="err">{error}</div>}
      {info && <div className="err" style={{ background: "var(--sage-wash)", borderColor: "#e2e6da", color: "var(--sage-deep)" }}>{info}</div>}

      <input
        className="field"
        type="email"
        placeholder="you@example.com"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        autoComplete="email"
      />
      <input
        className="field"
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        autoComplete={mode === "signin" ? "current-password" : "new-password"}
        onKeyDown={(e) => e.key === "Enter" && submit()}
      />

      <div className="actions" style={{ marginTop: 4 }}>
        <button className="btn-primary" onClick={submit} disabled={busy || !email || !password}>
          {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
        </button>
      </div>

      <p style={{ marginTop: 18, fontSize: 13.5, color: "var(--ink-soft)" }}>
        {mode === "signin" ? "New here? " : "Already have an account? "}
        <button className="muted-link" onClick={() => { setMode(mode === "signin" ? "signup" : "signin"); setError(null); }}>
          {mode === "signin" ? "Create an account" : "Sign in"}
        </button>
      </p>
    </div>
  );
}
