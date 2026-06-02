# ImmunoSense Web — first vertical slice

Real Vite + React + TypeScript app. This slice covers the core loop:
**Auth (Supabase) → Daily check-in → /v1/evaluate → Reflection**, wired to your
live API. The other screens (onboarding, history, settings) come next.

## Prerequisites
- Node 18+ (you have v24 — good)
- The ImmunoSense API running locally with CORS allowing the Vite origin

## 1. Configure environment
Copy the example env and fill it in:

    copy .env.example .env.local

`.env.local` (values you already have):

    VITE_SUPABASE_URL=https://vsqdjxamurvoxafuokct.supabase.co
    VITE_SUPABASE_ANON_KEY=sb_publishable_4E6XX1AhYmAmK1PoMwBzNw_On0z5PZM
    VITE_API_BASE_URL=http://127.0.0.1:8000

## 2. Start the API (in your "live" terminal, from the project root)

    set DATABASE_URL=postgresql://postgres:Strivehard_2026@db.vsqdjxamurvoxafuokct.supabase.co:5432/postgres
    set DEV_AUTH=0
    set SUPABASE_JWKS_URL=https://vsqdjxamurvoxafuokct.supabase.co/auth/v1/.well-known/jwks.json
    set CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
    venv\Scripts\python.exe -m uvicorn server.api.app:app --port 8000

## 3. Start the web app (in a separate terminal, in this folder)

    npm install
    npm run dev

Open http://localhost:5173

## What to test (the loop)
1. Sign in with the test user (test@immunosense.dev / Continue_2026), or create one.
2. Daily check-in: set energy, tap moods, optional note → Continue.
3. You should land on the Reflection screen with a confidence-aware result
   (likely the "Building / not enough data" state at first — that's correct).
4. Try the flare button — it logs + evaluates immediately.

## Notes
- The check-in maps the 1-5 energy scale to the backend's fatigue severity
  (energized = low fatigue). We'll refine this mapping as we iterate.
- Auth uses the Supabase JS client; it stores/refreshes the session and the API
  client attaches the JWT automatically.
- If calls fail with CORS errors in the browser console, confirm the API was
  started with CORS_ORIGINS including http://localhost:5173.
- If you see 401s, the token may have expired — sign out and back in.

## Verified before shipping
- `npm run build` succeeds (tsc strict + vite build, 78 modules).
- `npx tsc --noEmit` passes with no errors.
Browser rendering + live API calls are verified by YOU running it (the part a
sandbox can't check).
