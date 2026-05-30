"""Fixtures for server data-layer tests.

Each test gets a fresh in-memory SQLite DB with the full schema created from
the models. This exercises the same SQLAlchemy code path that runs on Postgres
in production, without needing a Postgres server in CI.
"""

import pytest
from sqlalchemy.orm import sessionmaker

from server.db.base import Base, make_engine
from server.db import models  # noqa: F401  (register models on Base.metadata)


@pytest.fixture
def session_factory():
    """Fresh in-memory SQLite DB + session factory per test."""
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
