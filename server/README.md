# ImmunoSense Server — Data Layer

The deployable service that wraps the `immunosense/` library. This package is
the data layer (Phase 1 of the server build); the API and auth layers come next.

## What's here

```
server/
├── db/
│   ├── base.py            engine/session, four-schema config, DATABASE_URL driven
│   ├── types.py           dialect-aware types + PHI encryption seam
│   ├── models.py          all tables: identity / health / reference / audit
│   ├── event_store.py     PostgresEventLog — drop-in for the NDJSON EventLog
│   ├── seed.py            loads LR table + corroboration patterns into reference schema
│   └── migrations/        Alembic (initial schema migration included)
├── builders/              (next step) raw input -> agent domain objects
├── api/                   (next step) FastAPI app + Supabase auth
├── tests/                 22 data-layer tests
└── requirements.txt
```

The dependency direction is one-way: **`server/` imports `immunosense/`, never
the reverse.** The library stays storage- and web-agnostic.

## Install deps

```cmd
venv\Scripts\python.exe -m pip install -r server\requirements.txt
```

## Local dev (SQLite — no Postgres needed)

The data layer runs on SQLite for local dev and tests. Nothing to set up:

```cmd
venv\Scripts\python.exe verify_datalayer.py
venv\Scripts\python.exe -m pytest server\tests -q
```

Expect `RESULT: 26/26 checks passed` and `22 passed`.

## Production (Supabase Postgres) — YOUR setup

The data layer is engine-agnostic via SQLAlchemy. To run against your Supabase
Postgres, set `DATABASE_URL` and run the migration:

```cmd
set DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
venv\Scripts\python.exe -m alembic upgrade head
venv\Scripts\python.exe -c "from sqlalchemy.orm import sessionmaker; from server.db.base import make_engine; from server.db.seed import seed_reference_data; import os; e=make_engine(os.environ['DATABASE_URL']); print(seed_reference_data(sessionmaker(bind=e, future=True)))"
```

The migration's `env.py` auto-creates the four schemas (`identity`, `health`,
`reference`, `audit`) on Postgres before creating tables.

### What is sandbox-verified vs needs your Supabase setup

**Verified in the sandbox (works identically on Postgres via SQLAlchemy):**
- All 16 tables + the migration (upgrade and downgrade)
- `PostgresEventLog` drop-in parity with the real Conductor
- The reference seeder
- The PHI encryption seam (passthrough in Phase 1)
- Audit logging on writes

**Needs your Supabase project (cannot be verified without your credentials):**
- The actual `DATABASE_URL` connection to Supabase
- **Supabase Auth ↔ `identity.users` linkage.** Supabase Auth manages its own
  `auth.users` table. In Phase 1 our `identity.users.user_id` should be the
  Supabase auth user's UUID. You'll wire this when the API/auth layer is built
  (next step) — the API validates the Supabase JWT, extracts the `sub` (the
  auth user id), and uses it as `user_id`.
- **Row-Level Security (RLS) policies.** For defence-in-depth you'll want RLS on
  the `health` schema so a user can only read their own rows. This is a Supabase
  dashboard / SQL step, documented when we build auth. Phase 1 can run without
  RLS (the API enforces access), but RLS is part of the HIPAA-ready posture.

## The PHI encryption seam

PHI columns use the `EncryptedString` type. In Phase 1 it's a **passthrough**
(stores plaintext) — but it MARKS every PHI column, so:
- tooling can enumerate PHI columns (`col.type.is_phi`)
- enabling encryption later is a config flip (set `IMMUNOSENSE_PHI_KEY` and
  implement `EncryptedString._encrypt/_decrypt`), with NO schema or model change

This is the "HIPAA-ready, not HIPAA-required in Phase 1" mechanism in concrete
form.

## What does NOT change in the library

The whole point of `PostgresEventLog`: the Conductor and all 752 library tests
are untouched. The Conductor consumes whatever `EventLog`-shaped object it's
given — NDJSON file store or Postgres. Swapping is one construction line:

```python
# before (file-based)
log = EventLog("data/events")
# after (Postgres)
log = PostgresEventLog(session_factory)
conductor = Conductor(registry=registry, event_log=log)  # unchanged
```

## Next step

The **builders** (raw UI input → agent domain objects) and the **API layer**
(FastAPI + Supabase auth). The data layer is the foundation those sit on.
