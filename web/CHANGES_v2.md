# Web app update v2 — meal logging + dev agent inspector

## New since the first slice
1. MEAL LOGGING in the check-in flow: "+ Add a meal" reveals a text field.
   The TEXT drives the dietary pipeline (POST /v1/log/meal). A photo is noted as
   "coming soon, record only" — no AI reads the photo (Phase 1 boundary).
2. DEV AGENT INSPECTOR: a small "⚙ dev: inspect agents" link (low-opacity,
   bottom of check-in/reflection). Opens a builder-only view showing per-agent
   ok/dim/confidence/quality, fusion, raw probability, patterns, warnings.
   Calls POST /v1/evaluate/debug.

## To use the inspector you MUST start the API with the debug flag
  set ENABLE_DEBUG_ENDPOINT=1
(plus your usual live settings + CORS). Without it, the inspector shows a clear
"Debug endpoint disabled" message (the endpoint 404s by design).

## Run (same as before)
  npm install   (only if you haven't, or after replacing files)
  npm run dev
Open http://localhost:5173

## Verified before shipping
  npx tsc --noEmit  -> clean
  npm run build     -> 79 modules, ok
Browser + live API verified by you.
