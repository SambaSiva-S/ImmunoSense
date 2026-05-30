# ImmunoSense API

The FastAPI service that ties the data layer, builders, and Conductor together
behind HTTP endpoints. Phase 1 surface: log, evaluate, report.

## Endpoints

```
GET  /health                 liveness
POST /v1/log/symptom         log a symptom entry
POST /v1/log/meal            log a meal (text + optional photo_id)
POST /v1/log/biomarker       log a self-entered lab reading
POST /v1/log/flare           flare button — logs AND evaluates immediately
POST /v1/evaluate            evaluate the current bucket (normal path)
GET  /v1/report/latest       most recent bucket report
GET  /v1/history             past reports (paginated)
GET  /v1/me                  profile + consent state
PUT  /v1/me/consent          grant/revoke a consent type
POST /v1/photo               signed upload URL for a food photo
```

`user_id` is always derived from auth, never from the request body.

## Run locally (dev mode — no Supabase needed)

```cmd
venv\Scripts\python.exe -m pip install -r server\requirements.txt
set DEV_AUTH=1
venv\Scripts\python.exe -m uvicorn server.api.app:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API docs. In dev mode,
authenticate by sending an `X-Dev-User: <some-id>` header instead of a JWT.

Quick smoke test:
```cmd
venv\Scripts\python.exe verify_api.py
venv\Scripts\python.exe -m pytest server\tests\test_api.py -q
```
Expect `RESULT: 19/19` and `20 passed`.

## Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | sqlite dev file | Postgres/Supabase URL in prod |
| `DEV_AUTH` | `1` | `0` in prod to require real JWT |
| `SUPABASE_JWKS_URL` | — | Supabase project's JWKS endpoint (prod) |
| `SUPABASE_JWT_AUD` | `authenticated` | JWT audience claim |
| `DIETARY_DENSITY_CACHE` | — | path to density.pkl (enables dietary) |
| `DIETARY_FOOD_INDEX_CACHE` | — | path to food_index.pkl |
| `USE_CLAUDE_TFM` | `0` | `1` to call live ClaudeTFM for explanations |
| `DEFAULT_DISEASE` | `SLE` | fallback when a profile has no disease |

## Tracelog / error tracking

Every request gets a trace id (from the `X-Trace-Id` request header or
generated). It's returned in the `X-Trace-Id` response header and propagated
into the Conductor's Layer A events, so a bucket evaluation can be traced back
to the HTTP request. Requests and unhandled errors are logged as structured
records (`grep` the logs by trace id to follow one request). Unhandled
exceptions are caught and returned as a clean JSON 500 with the trace id —
never a leaked stack trace.

## Production auth — Supabase JWT (STEP 5, your setup)

Set `DEV_AUTH=0` and provide `SUPABASE_JWKS_URL`. Then:

- The client sends `Authorization: Bearer <supabase-jwt>`.
- The API validates the JWT against Supabase's JWKS, extracts the `sub` claim
  (the Supabase auth user id) — that is the `user_id`.

The JWKS URL is typically:
```
https://<project-ref>.supabase.co/auth/v1/.well-known/jwks.json
```
(or the legacy `/auth/v1/keys`). Confirm in your Supabase dashboard under
Project Settings → API.

### Verifying real auth (do this together)

1. Create a test user in Supabase Auth.
2. Get a JWT for that user (sign in via the Supabase client, or the dashboard).
3. Call the API with `Authorization: Bearer <jwt>` and `DEV_AUTH=0`.
4. Confirm the request succeeds and operates on the user id from the token.

### Row-Level Security (RLS)

The API enforces per-user access (every query filters by the authed user_id).
For defence-in-depth, also enable RLS on the `health` schema in Supabase so the
database itself rejects cross-user reads. This is a Supabase SQL step; Phase 1
can run without it (the API enforces access), but it's part of the HIPAA-ready
posture for the clinical upgrade.

## Photo upload (STEP 5, your setup)

`POST /v1/photo` returns a signed upload URL + a `photo_id`. In dev mode the URL
is a stub. In production, wire it to Supabase Storage:
- issue a real signed upload URL (Supabase Storage `createSignedUploadUrl`)
- the storage edge function strips EXIF on upload
- the client uploads bytes directly to storage, then attaches `photo_id` to a
  `/v1/log/meal` call

## What's sandbox-verified vs needs your Supabase

**Verified here (dev mode):** all endpoints, the confidence-aware framing,
tracelog, audit logging, user isolation, the full log→evaluate→report loop.

**Needs your Supabase:** real JWT validation (the JWKS code is written but only
testable against your project), and the real photo signed-URL issuance. Both are
documented above; we verify them together against your live project.
