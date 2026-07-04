# Deploying the ImmunoSense API — step-by-step (audited)

Deploys the backend API to a public URL your mobile app (and web app) can reach.
Supabase (DB + auth + storage) is already hosted — this is just the FastAPI service.

Recommended host: **Render** (free tier, deploys from GitHub). The Procfile also
works on Railway/Heroku.

---

## DEPLOYMENT READINESS AUDIT — results

Checked, and PASSING:
- [x] App boots via `uvicorn server.api.app:app` (the exact host start command)
- [x] `/health` returns 200 (Render's health check hits this — verified live)
- [x] Reads `$PORT` from the host (via the Procfile/start command)
- [x] `server/` and `immunosense/` are proper importable packages
- [x] No hardcoded local file paths in the API that would break on the host
- [x] Model files absent is handled gracefully — biomarker/dietary run degraded,
      app still boots + evaluates (same as the passing test suite)
- [x] DII artifact .pkl files ARE tracked in git (despite .gitignore) -> on host
- [x] All server + agent imports are covered by requirements.txt
- [x] anthropic is lazy-imported only (optional); included anyway to be safe
- [x] CORS is env-driven (CORS_ORIGINS) — set it to your web/mobile origins
- [x] No REQUIRED env vars crash the app if missing (all have defaults) — BUT see
      the warning below about DATABASE_URL

Watch-outs (not blockers, but know them):
- [!] **torch is ~800MB-2GB.** It's required (biomarker imports it at load). On
      Render's FREE tier this may exceed memory at build/boot. If deploy fails with
      OOM/disk -> upgrade to **Starter ($7/mo)**. (Confirmed big: it filled the
      build sandbox's disk here.)
- [!] **Missing DATABASE_URL silently falls back to SQLite** (no crash). So if the
      app boots but has no data, check that DATABASE_URL is actually set on the host.
- [!] **Migrations don't auto-run on startup.** Your Supabase DB is already migrated
      (through d4e5f6a7b8c9), so this is fine. If you add migrations later, run
      `alembic upgrade head` against Supabase before/after deploy.

---

## Files in the repo for deployment
- `requirements.txt` — runtime deps (audited complete)
- `Procfile` — start command
- `runtime.txt` — pins Python 3.12.3
- `render.yaml` — Render service config

Commit them:
    git add requirements.txt Procfile runtime.txt render.yaml DEPLOY_NOTES.md
    git commit -m "Add deployment config (audited)"
    git push

---

## Step 1 — Create the Render service
1. https://render.com -> sign up/log in (GitHub works).
2. New + -> Web Service -> connect GitHub -> pick the **ImmunoSense** repo.
3. If it doesn't auto-read render.yaml, set manually:
   - Runtime: Python 3
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn server.api.app:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`

## Step 2 — Environment variables (Render dashboard -> Environment)
Use the POOLER hosts (the direct db host times out on IPv6):

    DATABASE_URL            postgresql://postgres.vsqdjxamurvoxafuokct:<pw>@aws-1-us-west-2.pooler.supabase.com:5432/postgres
    APP_DATABASE_URL        postgresql://immunosense_app.vsqdjxamurvoxafuokct:<pw>@aws-1-us-west-2.pooler.supabase.com:5432/postgres
    SUPABASE_JWKS_URL       https://vsqdjxamurvoxafuokct.supabase.co/auth/v1/.well-known/jwks.json
    SUPABASE_URL            https://vsqdjxamurvoxafuokct.supabase.co
    SUPABASE_SERVICE_ROLE_KEY   <service_role key>
    SUPABASE_STORAGE_BUCKET meal-photos
    CORS_ORIGINS            https://<your-web-app-url>
    ENABLE_DEBUG_ENDPOINT   0
    DEV_AUTH                0
    PYTHON_VERSION          3.12.3

## Step 3 — Deploy & watch the logs
Create Web Service. Watch the build + boot logs.
- Success: `Uvicorn running`, health check passes, you get a public URL.
- If build fails on torch (OOM) -> upgrade plan to Starter, redeploy.

## Step 4 — Verify
- `https://<render-url>/health` -> {"status":"ok"}
- `https://<render-url>/docs` -> FastAPI interactive docs

## Step 5 — Wire the web app (optional now)
Point web/.env's API base at the Render URL; add the web app's URL to CORS_ORIGINS.

---

## When it fails (first deploys often do) — quick diagnosis
- Build OOM/disk on torch          -> upgrade to Starter plan
- /health 500                       -> DATABASE_URL / APP_DATABASE_URL wrong
- App has no data / uses SQLite     -> DATABASE_URL not set on host
- CORS errors from web app          -> add web URL to CORS_ORIGINS
- Auth fails (401 on valid token)   -> SUPABASE_JWKS_URL wrong
Paste the Render logs and we'll debug together.

---

## SECURITY NOTE (honest, once)
Deploying puts the backend + its secrets on the public internet. The secrets
exposed in development become reachable by anyone who finds the endpoint. You're
setting fresh env vars on Render anyway, so this is the low-effort moment to
rotate them. Your call — but "later" gets real once the API is public.
