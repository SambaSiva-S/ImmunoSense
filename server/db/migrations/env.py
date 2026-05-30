"""Alembic migration environment.

Resolves the DB URL from DATABASE_URL at runtime (Supabase Postgres in prod,
SQLite for dev/tests), creates the four Postgres schemas before running
migrations, and targets the server models' metadata for autogenerate.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import text

from server.db.base import (
    SCHEMA_AUDIT,
    SCHEMA_HEALTH,
    SCHEMA_IDENTITY,
    SCHEMA_REFERENCE,
    Base,
    get_database_url,
    make_engine,
)

# Import all models so they register on Base.metadata for autogenerate.
from server.db import models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _ensure_schemas(connection) -> None:
    """Create the four schemas on Postgres (no-op on SQLite)."""
    if connection.dialect.name != "postgresql":
        return
    for schema in (SCHEMA_IDENTITY, SCHEMA_HEALTH, SCHEMA_REFERENCE, SCHEMA_AUDIT):
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def run_migrations_offline() -> None:
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(get_database_url())
    with engine.connect() as connection:
        _ensure_schemas(connection)
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
