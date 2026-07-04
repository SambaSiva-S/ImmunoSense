"""add home location to profiles (Environment agent Phase 1)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-03

Adds home_lat / home_lng / home_label to identity.profiles so the Environment
agent can look up air quality, pollen, etc. for the user's location. Nullable —
existing users simply have no location until they set one.
"""
from alembic import op
import sqlalchemy as sa


revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None

SCHEMA = "identity"


def _schema_kw():
    # On SQLite (tests) there are no schemas; pass schema=None.
    bind = op.get_bind()
    return {} if bind.dialect.name == "sqlite" else {"schema": SCHEMA}


def upgrade() -> None:
    kw = _schema_kw()
    op.add_column("profiles", sa.Column("home_lat", sa.Float(), nullable=True), **kw)
    op.add_column("profiles", sa.Column("home_lng", sa.Float(), nullable=True), **kw)
    op.add_column("profiles", sa.Column("home_label", sa.String(length=120), nullable=True), **kw)


def downgrade() -> None:
    kw = _schema_kw()
    op.drop_column("profiles", "home_label", **kw)
    op.drop_column("profiles", "home_lng", **kw)
    op.drop_column("profiles", "home_lat", **kw)
