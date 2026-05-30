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


@pytest.fixture
def api_client():
    """A TestClient backed by a shared in-memory SQLite DB (StaticPool) + dev auth."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    from server.api.app import create_app
    from server.api.config import Settings

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    app = create_app(session_factory=sf, settings=Settings(dev_auth=True))
    client = TestClient(app)
    client._session_factory = sf
    return client


@pytest.fixture
def auth_headers():
    return {"X-Dev-User": "u_test"}
