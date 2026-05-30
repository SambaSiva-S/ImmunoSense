"""SQLAlchemy declarative base, schema naming, and session management.

Schemas (Postgres): identity, health, reference, audit (D8 locked).
On SQLite (tests), schemas aren't supported the same way, so we map them to
table-name prefixes via a naming helper — the models stay identical, only the
physical layout differs per engine.

Connection: production points at Supabase Postgres via DATABASE_URL. Tests use
an in-memory or temp-file SQLite. The engine/session are created from the URL
so nothing in the models is engine-specific.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Schema names (Postgres). On SQLite these become None and tables live in the
# default schema (SQLite has no schema concept the same way).
SCHEMA_IDENTITY = "identity"
SCHEMA_HEALTH = "health"
SCHEMA_REFERENCE = "reference"
SCHEMA_AUDIT = "audit"


def _use_schemas() -> bool:
    """Whether to use real DB schemas (Postgres) vs flat (SQLite tests)."""
    url = os.environ.get("DATABASE_URL", "")
    return url.startswith("postgresql")


class Base(DeclarativeBase):
    """Shared declarative base for all ImmunoSense server models."""

    # Consistent naming convention so Alembic autogenerate is stable.
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


def schema_for(name: str):
    """Return the schema name on Postgres, or None on SQLite.

    Usage in models: __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}
    """
    return name if _use_schemas() else None


# --------------------------------------------------------------------------- #
# Engine + session
# --------------------------------------------------------------------------- #
def get_database_url() -> str:
    """Resolve the DB URL. Defaults to a local SQLite file for dev/tests."""
    return os.environ.get("DATABASE_URL", "sqlite:///immunosense_dev.db")


def make_engine(url: str | None = None, echo: bool = False):
    """Create a SQLAlchemy engine for the given URL (or the resolved default)."""
    url = url or get_database_url()
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=echo, future=True, connect_args=connect_args)


_engine = None
_SessionFactory = None


def init_engine(url: str | None = None, echo: bool = False):
    """Initialize the module-level engine + session factory."""
    global _engine, _SessionFactory
    _engine = make_engine(url, echo=echo)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def get_engine():
    if _engine is None:
        init_engine()
    return _engine


@contextmanager
def session_scope():
    """Transactional session scope. Commits on success, rolls back on error."""
    if _SessionFactory is None:
        init_engine()
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
