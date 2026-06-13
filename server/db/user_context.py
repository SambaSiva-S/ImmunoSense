"""Per-request database user context for Row-Level Security (Path 2).

ROBUST design (connection-level): a request sets the current user once, in a
ContextVar. A SQLAlchemy event then sets the Postgres GUC ``app.current_user_id``
at the start of EVERY transaction on EVERY connection — so all sessions (routes,
the EvaluationService, the conductor's event log, the scheduler) are covered
automatically, with no per-call-site effort. This fixes the gap where internal
sessions bypassed the context and broke under RLS.

On SQLite (tests/dev) everything is a no-op (no GUCs, no RLS).

Usage:
  - set_current_user(uid) at request start (and reset at end), OR
  - with user_session(sf, uid) as s:  (sets the ContextVar for that block)
Both work; the engine event does the actual GUC setting.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

# Request-scoped current user. ContextVar is safe across threads/async and per
# request (FastAPI runs each request in its own context).
_current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "immunosense_current_user", default=None
)


def set_current_user(user_id: str | None) -> contextvars.Token:
    """Set the current user for this context. Returns a token to reset with."""
    return _current_user.set(user_id)


def reset_current_user(token: contextvars.Token) -> None:
    _current_user.reset(token)


def get_current_user() -> str | None:
    return _current_user.get()


def install_rls_hook(engine: Engine) -> None:
    """Install a transaction-start hook on a Postgres engine that sets
    app.current_user_id from the ContextVar. No-op for non-Postgres engines.

    Idempotent: safe to call once per engine at startup.
    """
    if engine.dialect.name != "postgresql":
        return
    if getattr(engine, "_rls_hook_installed", False):
        return

    @event.listens_for(engine, "begin")
    def _set_user_on_begin(conn):
        uid = _current_user.get()
        # set_config(..., true) = transaction-local; resets at COMMIT/ROLLBACK so
        # it never leaks across pooled connections.
        conn.exec_driver_sql(
            "SELECT set_config('app.current_user_id', %s, true)",
            (uid if uid is not None else "",),
        )

    engine._rls_hook_installed = True  # type: ignore[attr-defined]


def _is_postgres(session: Session) -> bool:
    return session.bind is not None and session.bind.dialect.name == "postgresql"


@contextmanager
def user_session(session_factory: sessionmaker, user_id: str | None):
    """Open a session with the current user set for its duration.

    Sets the ContextVar (which the engine 'begin' hook reads) so the GUC is
    applied to this session's transactions. Also still works if the engine hook
    isn't installed, by setting the GUC directly as a fallback.
    """
    token = _current_user.set(user_id)
    s = session_factory()
    try:
        # Fallback: if no engine hook is installed, set it directly so behavior
        # is correct either way. (Harmless if the hook also sets it.)
        if user_id is not None and _is_postgres(s):
            s.execute(
                text("SELECT set_config('app.current_user_id', :uid, true)"),
                {"uid": str(user_id)},
            )
        yield s
    finally:
        s.close()
        _current_user.reset(token)


def current_db_user(session: Session) -> str | None:
    """Read back the per-request user context (for tests/diagnostics)."""
    if not _is_postgres(session):
        return None
    row = session.execute(
        text("SELECT current_setting('app.current_user_id', true)")
    ).scalar()
    return row or None
