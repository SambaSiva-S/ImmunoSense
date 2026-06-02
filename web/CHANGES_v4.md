# Web app v4 — History/Trends + Settings (+ bottom nav)

## New since v3
1. HISTORY / TRENDS screen: a sparkline of your severity trend over time + a
   list of past reflections (each with its confidence pill). Sparse-data
   graceful: 0 reflections -> "your history is just beginning"; 1 point ->
   "a line appears with a couple more check-ins"; 2+ -> the trend line draws.
   Uses GET /v1/history (existing — no backend change).
2. SETTINGS screen: profile (email + condition), two consent toggles
   (AI explanations, research) wired to PUT /v1/me/consent, and Sign out.
   Uses GET /v1/me (existing).
3. BOTTOM NAV: Today / History / Settings across the main sections.

## Frontend-only — no backend changes
All endpoints already exist. Just replace the web source and re-run.

## Run
  npm install   (no new deps; safe to skip if unchanged)
  npm run dev

## Verify
- Bottom nav switches Today / History / Settings.
- History: after your existing check-ins, you'll see past reflection rows; the
  trend line needs 2+ days of data to draw (you have ~1 bucket so far, so expect
  the "one data point" note — that's correct).
- Settings: toggles flip and persist (re-open Settings to confirm they stuck);
  Sign out works.

## Built/verified here
  npx tsc --noEmit -> clean ; npm run build -> 375 kB bundle, ok
Browser + live API verified by you.
