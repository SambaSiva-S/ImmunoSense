"""enable Row-Level Security: app role + per-user policies

Revision ID: c3d4e5f6a7b8
Revises: a1b2c3d4e5f6
Create Date: 2026-06-13

Path-2 RLS. Creates a NON-superuser role `immunosense_app` and per-user RLS
policies on every user-keyed table, keyed off the session variable
`app.current_user_id` (set per-request by server/db/user_context.py).

IMPORTANT: this migration does NOT change who the API connects as. The policies
exist but a superuser (current API connection) still bypasses them. Increment 3
switches the API to connect as `immunosense_app`, at which point the database
enforces per-user isolation. So applying this migration alone is non-breaking.

Postgres-only. On SQLite (tests) this is a no-op.
"""
from alembic import op
import sqlalchemy as sa


revision = "c3d4e5f6a7b8"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None

# (schema, table) for every user-keyed table. reference.* excluded (shared).
USER_TABLES = [
    ("identity", "users"),
    ("identity", "profiles"),
    ("identity", "consents"),
    ("health", "events"),
    ("health", "bucket_reports"),
    ("health", "symptom_logs"),
    ("health", "dietary_logs"),
    ("health", "biomarker_readings"),
    ("health", "wearable_readings"),
    ("health", "photos"),
    ("health", "flare_button_events"),
    ("audit", "access_log"),
    ("audit", "delete_log"),
]

# Role the API will connect as (Increment 3). Created without LOGIN here; the
# password/login is set out-of-band (we set it on Supabase during apply) so no
# secret lives in the migration.
APP_ROLE = "immunosense_app"

# users keys on user_id as PK (the user's own identity row); audit tables allow
# NULL user_id (system events) — those rows are visible to no app user, which is
# correct (only the service role reads full audit).
NULLABLE_USER_ID = {("audit", "access_log"), ("audit", "delete_log")}


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return  # no RLS on sqlite; tests exercise policy logic against Postgres

    # 1. create the app role if absent (NOLOGIN until Increment 3 sets a password)
    op.execute(f"""
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
        CREATE ROLE {APP_ROLE} NOLOGIN;
      END IF;
    END $$;
    """)

    # 2. let the app role use the schemas
    for schema in ("identity", "health", "reference", "audit"):
        op.execute(f'GRANT USAGE ON SCHEMA "{schema}" TO {APP_ROLE};')

    # reference.* is shared read-only lookup data
    op.execute(f'GRANT SELECT ON ALL TABLES IN SCHEMA "reference" TO {APP_ROLE};')

    # 3. per-table: grant DML, enable + force RLS, add the per-user policy
    for schema, table in USER_TABLES:
        fq = f'"{schema}"."{table}"'
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {fq} TO {APP_ROLE};")
        op.execute(f"ALTER TABLE {fq} ENABLE ROW LEVEL SECURITY;")
        # FORCE so even the table owner is subject to RLS (defense in depth).
        op.execute(f"ALTER TABLE {fq} FORCE ROW LEVEL SECURITY;")

        if (schema, table) in NULLABLE_USER_ID:
            # audit rows: a user sees only their own; NULL (system) rows hidden.
            using = "user_id = current_setting('app.current_user_id', true)"
            check = using
        else:
            using = "user_id = current_setting('app.current_user_id', true)"
            check = using

        op.execute(f"""
        CREATE POLICY {table}_isolation ON {fq}
          USING ({using})
          WITH CHECK ({check});
        """)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    for schema, table in USER_TABLES:
        fq = f'"{schema}"."{table}"'
        op.execute(f"DROP POLICY IF EXISTS {table}_isolation ON {fq};")
        op.execute(f"ALTER TABLE {fq} NO FORCE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {fq} DISABLE ROW LEVEL SECURITY;")
        op.execute(f"REVOKE ALL ON {fq} FROM {APP_ROLE};")
    for schema in ("identity", "health", "reference", "audit"):
        op.execute(f'REVOKE USAGE ON SCHEMA "{schema}" FROM {APP_ROLE};')
    op.execute(f'REVOKE SELECT ON ALL TABLES IN SCHEMA "reference" FROM {APP_ROLE};')
    # Drop the role last (after privileges revoked).
    op.execute(f"DROP ROLE IF EXISTS {APP_ROLE};")
