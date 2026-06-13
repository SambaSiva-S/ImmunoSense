"""Per-request database user context for Row-Level Security (Path 2).

This is the plumbing RLS relies on: each request opens its DB session and sets a
session-local variable ``app.current_user_id`` to the authenticated user. RLS
policies (added in a later increment) compare each row's ``user_id`` against this
value, so the database itself enforces per-user isolation — even if application
code forgets to filter.

Increment 1 scope: this helper SETS the context but no policies READ it yet, so
behavior is unchanged. It is safe on SQLite (tests/dev), where it is a no-op
because SQLite has no RLS or session GUCs.

Usage in a route (replacing ``with sf() as s:``)::

    with user_session(sf, user_id) as s:
        ...  # queries here run with app.current_user_id set
"""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, Session


def _is_postgres(session: Session) -> bool:
    return session.bind is not None and session.bind.dialect.name == "postgresql"


@contextmanager
def user_session(session_factory: sessionmaker, user_id: str | None):
    """Open a session with the RLS user-context set for this request.

    On Postgres, runs ``SELECT set_config('app.current_user_id', :uid, true)``
    which sets the value for the duration of the transaction (the ``true`` makes
    it transaction-local, so it never leaks across pooled connections). On SQLite
    it is a no-op. Always yields a normal session usable exactly like
    ``with session_factory() as s``.
    """
    s = session_factory()
    try:
        if user_id is not None and _is_postgres(s):
            # transaction-local (is_local=true) so it resets at COMMIT/ROLLBACK
            # and never bleeds onto the next user via a reused pooled connection.
            s.execute(
                text("SELECT set_config('app.current_user_id', :uid, true)"),
                {"uid": str(user_id)},
            )
        yield s
    finally:
        s.close()


def current_db_user(session: Session) -> str | None:
    """Read back the per-request user context (for tests/diagnostics)."""
    if not _is_postgres(session):
        return None
    row = session.execute(
        text("SELECT current_setting('app.current_user_id', true)")
    ).scalar()
    return row or None
