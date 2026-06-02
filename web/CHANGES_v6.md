# Web app v6 — demographics in onboarding (age/sex/height/weight -> BMI)

## New since v5
Onboarding is now 2 steps:
  Step 1: condition (as before)
  Step 2: sex, date of birth, height, weight — with DUAL UNITS:
    height in cm OR ft/in; weight in kg OR lb. The app converts to canonical
    cm/kg and sends those; BMI is computed (shown as a live preview).
These populate the demographics the biomarker + dietary agents need for their
percentile baselines (previously defaulted to 40/1/25 for everyone — a bug the
backend increment fixes).

## REQUIRES the backend demographics increment + MIGRATION
Apply immunosense_demographics.tar.gz AND run the alembic migration first
(see its notes), or saving step 2 will error on the new columns.

## Run
  npm install
  npm run dev

## Verify
- A user with no condition sees Onboarding. Step 2 collects demographics; toggle
  cm<->ft/in and kg<->lb — BMI preview updates.
- "Get started" saves; the API log stops warning about missing demographics.
- Inspector: agent1_biomarker now runs with your real age/sex/BMI.

## Built/verified here: tsc clean, vite build ok.
