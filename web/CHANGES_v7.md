# Web app v7 — biomarker backfill (enter past lab dates)

## New since v6
The biomarker screen now has a "When was this taken?" date field (defaults to
today, capped at today). Enter older bloodwork with its real draw date — past
results land in the right historical bucket and help build your baseline faster.

## REQUIRES backend immunosense_biomarker_backfill.tar.gz (no migration).

## Run
  npm install ; npm run dev

## Verify
- Add a lab result, set the date to a past date (e.g. last month), CRP 8 -> Save.
- It logs to that historical date (visible in the API tracelog bucket id).
- Future dates are blocked by the date picker (max=today) and the server (422).

## Built/verified here: tsc clean, vite build ok.
