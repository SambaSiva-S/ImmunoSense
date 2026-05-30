# API layer — install notes

## What's in this package (additive + two updates)

NEW:
  server/api/                  FastAPI app, routes, auth, service, schemas, tracelog, config
  server/tests/test_api.py     20 API tests
  verify_api.py                19-check API verifier
  server/api/README.md         run + Supabase wiring guide

UPDATED (replace existing):
  server/db/models.py          user_id is now String(128) everywhere (was GUID on
                               identity tables) — a consistency fix the API testing
                               surfaced. Supabase user ids are UUIDs stored as strings.
  server/tests/conftest.py     adds api_client + auth_headers fixtures
  server/tests/test_models_and_seed.py   two tests use str(uuid4()) now
  server/requirements.txt      adds fastapi/uvicorn/pydantic/pyjwt/pandas/pyreadstat
  server/db/migrations/versions/bb6b63ad1b79_initial_schema.py
                               REGENERATED migration (replaces d135a14ecda2) for the
                               user_id type fix.

## IMPORTANT — the migration was regenerated

The old migration file `d135a14ecda2_initial_schema.py` is REPLACED by
`bb6b63ad1b79_initial_schema.py`. Delete the old one:

    del server\db\migrations\versions\d135a14ecda2_initial_schema.py

Because you have NOT yet run the migration against a real Supabase DB (we're
still on SQLite dev), this is a clean swap — no data migration needed. On SQLite
dev the tables are created from the models directly, so nothing to undo.

## Install

    venv\Scripts\python.exe -m pip install -r server\requirements.txt
    del server\db\migrations\versions\d135a14ecda2_initial_schema.py
    venv\Scripts\python.exe verify_api.py
    venv\Scripts\python.exe -m pytest server\tests -q

Expect: RESULT 19/19, and 57 passed (22 data layer + 15 builders + 20 API).

## Run the API

    set DEV_AUTH=1
    venv\Scripts\python.exe -m uvicorn server.api.app:app --reload
    # open http://127.0.0.1:8000/docs

## Next (STEP 5 — together, against your Supabase)

See server/api/README.md "Production auth" — create the project, set
DATABASE_URL + SUPABASE_JWKS_URL, run the migration against real Postgres,
verify a real JWT round-trip, enable RLS on the health schema.
