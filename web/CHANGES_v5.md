# Web app v5 — Onboarding + Biomarker entry

## New since v4
1. ONBOARDING (first run): if your profile has no condition, you're greeted with
   a setup screen — pick your condition + auto-detected timezone -> saved via
   PUT /v1/me/profile. "Skip for now" is available. After this, you go straight
   to the check-in. Returning users with a condition skip onboarding entirely.
2. BIOMARKER ENTRY: "+ Add a lab result" (under the check-in) opens a screen to
   enter CRP / ESR from your bloodwork -> POST /v1/log/biomarker. These feed the
   biomarker agent on your next evaluation.

## REQUIRES the backend profile endpoint
Apply immunosense_profile_endpoint.tar.gz first (adds PUT /v1/me/profile) and
restart the API, or onboarding's "Get started" will error.

## Run
  npm install   (no new deps)
  npm run dev

## Verify
- First sign-in with a fresh user (or one with no condition set) shows
  Onboarding. Pick a condition -> Get started -> lands on check-in.
- Settings should now show the condition you picked.
- "+ Add a lab result" -> enter CRP e.g. 8 -> Save -> "Saved" confirmation.
  Then run the inspector: agent1_biomarker should report.

## Built/verified here
  npx tsc --noEmit -> clean ; npm run build -> ok
