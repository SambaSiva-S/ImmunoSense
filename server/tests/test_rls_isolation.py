"""Cross-user RLS isolation test.

Runs ONLY when a real Postgres test URL is provided via RLS_TEST_PG_URL (a
superuser URL to an empty test DB). Skipped otherwise (the default SQLite suite
has no RLS). This guards the core security property: as the restricted role,
a user can read/write ONLY their own rows.

To run locally against Postgres:
  set RLS_TEST_PG_URL=postgresql+psycopg2://postgres@127.0.0.1:5433/rlstest
  pytest server/tests/test_rls_isolation.py -v
"""
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

PG = os.environ.get("RLS_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="RLS_TEST_PG_URL not set (Postgres-only test)")

APP = "immunosense_app"
APP_PW = "rls_test_pw"


def _app_url():
    # derive the app-role URL from the superuser URL
    after = PG.split("://", 1)[1]
    hostpart = after.split("@", 1)[1]
    return f"postgresql+psycopg2://{APP}:{APP_PW}@{hostpart}"


@pytest.fixture(scope="module")
def setup_db():
    # Create just the one table we test (symptom_logs) with explicit schema via
    # raw SQL, so the test doesn't depend on model import-order schema binding.
    sup = create_engine(PG, future=True)
    with sup.begin() as c:
        for s in ("identity", "health", "reference", "audit"):
            c.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{s}"'))
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS health.symptom_logs (
                log_id uuid PRIMARY KEY,
                user_id varchar(128) NOT NULL,
                bucket_id varchar(128),
                logged_at timestamptz,
                source varchar(32)
            )"""))
        c.execute(text(f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='{APP}') THEN CREATE ROLE {APP} NOLOGIN; END IF; END $$;"))
        c.execute(text(f'GRANT USAGE ON SCHEMA "health" TO {APP};'))
        fq = '"health"."symptom_logs"'
        c.execute(text(f"GRANT SELECT,INSERT,UPDATE,DELETE ON {fq} TO {APP};"))
        c.execute(text(f"ALTER TABLE {fq} ENABLE ROW LEVEL SECURITY;"))
        c.execute(text(f"ALTER TABLE {fq} FORCE ROW LEVEL SECURITY;"))
        c.execute(text(f"DROP POLICY IF EXISTS symptom_logs_isolation ON {fq};"))
        c.execute(text(f"CREATE POLICY symptom_logs_isolation ON {fq} USING (user_id=current_setting('app.current_user_id',true)) WITH CHECK (user_id=current_setting('app.current_user_id',true));"))
        c.execute(text(f"ALTER ROLE {APP} WITH LOGIN PASSWORD '{APP_PW}';"))
        for uid in ("alice", "bob"):
            c.execute(text("INSERT INTO health.symptom_logs (log_id,user_id,bucket_id,logged_at,source) VALUES (:i,:u,:b,:t,'tap')"),
                      {"i": str(uuid.uuid4()), "u": uid, "b": "b", "t": datetime.now(timezone.utc)})
    yield


def _run(sf, uid, sql, params=None):
    from server.db.user_context import user_session
    with user_session(sf, uid) as s:
        return s.execute(text(sql), params or {}).fetchall()


def test_read_isolation(setup_db):
    sf = sessionmaker(bind=create_engine(_app_url(), future=True), expire_on_commit=False, future=True)
    a = _run(sf, "alice", "SELECT user_id FROM health.symptom_logs")
    b = _run(sf, "bob", "SELECT user_id FROM health.symptom_logs")
    assert [r[0] for r in a] == ["alice"]
    assert [r[0] for r in b] == ["bob"]


def test_write_isolation(setup_db):
    from server.db.user_context import user_session
    sf = sessionmaker(bind=create_engine(_app_url(), future=True), expire_on_commit=False, future=True)
    with pytest.raises(Exception):
        with user_session(sf, "bob") as s:
            s.execute(text("INSERT INTO health.symptom_logs (log_id,user_id,bucket_id,logged_at,source) VALUES (:i,'alice','x',:t,'tap')"),
                      {"i": str(uuid.uuid4()), "t": datetime.now(timezone.utc)})
            s.commit()
