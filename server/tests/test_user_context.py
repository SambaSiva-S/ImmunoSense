"""Tests for the RLS per-request user-context helper (Increment 1)."""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from server.db.user_context import user_session, current_db_user


def _sqlite_sf():
    eng = create_engine("sqlite://", future=True)
    return sessionmaker(bind=eng, future=True)


class TestUserContext:
    def test_sqlite_is_noop_and_session_usable(self):
        sf = _sqlite_sf()
        with user_session(sf, "user_x") as s:
            assert s.execute(text("SELECT 1")).scalar() == 1
            # SQLite has no GUCs — helper must no-op, not error.
            assert current_db_user(s) is None

    def test_none_user_is_safe(self):
        sf = _sqlite_sf()
        with user_session(sf, None) as s:
            assert s.execute(text("SELECT 1")).scalar() == 1

    def test_session_closes_cleanly(self):
        sf = _sqlite_sf()
        with user_session(sf, "user_y") as s:
            pass
        # no exception on exit = pass
