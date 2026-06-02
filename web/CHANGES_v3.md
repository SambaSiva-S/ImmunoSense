# Web app v3 — enriched symptom check-in

## What changed
The check-in's first step now collects 5 structured symptom fields as gentle
1-5 taps (Fatigue, Joint pain, Sleep trouble, Brain fog, Gut/digestion) instead
of a single energy slider. Each is optional (tap again to clear). These map to
the real SymptomLogIn fields (fatigue, joint_pain, sleep_severity,
brain_fog_severity, gi_distress), so agent5_symptoms_mood gets enough structured
signal to report real confidence.

REQUIRES the backend builder fix (immunosense_confidence_fix.tar.gz) — without
it the agent still reports 0 confidence regardless of fields.

## Run
  npm install   (if needed)
  npm run dev

## Verify in the inspector
After logging several symptom taps + a meal, "inspect agents" should show
agent5_symptoms_mood with quality > 0 (rising with how many fields you tap),
alongside agent2_dietary. With enough signal the overall result moves off
"insufficient".

## Verified before shipping
  npx tsc --noEmit -> clean ; npm run build -> ok
