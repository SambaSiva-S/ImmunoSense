# Web app v8 — meal photo upload (record-only) + client-side compression

## New
- Meal section now has "📷 Attach a photo (optional)". It:
  1. compresses/resizes the image client-side (longest edge 1600px, JPEG ~0.82)
     so a multi-MB phone photo becomes a few hundred KB — crisp for your record,
     cheap to store. Falls back to original if compression isn't possible.
  2. requests a signed upload URL from /v1/photo
  3. uploads bytes DIRECTLY to Supabase Storage (never through our API)
  4. attaches the photo_id to the meal log
- Copy is explicit: the photo "isn't analyzed" — record-only (Option A). Text
  still drives dietary scoring. (AI photo analysis = Option B, parked.)

## REQUIRES backend immunosense_photo_storage.tar.gz + the new SUPABASE_* env
vars on the API (see its notes). In dev-stub mode the attach still works (skips
the real PUT).

## Run: npm install ; npm run dev
## Built/verified here: tsc clean, vite build ok.
