"""FastAPI dependency providers.

The session factory and service are created once at app startup and stored on
app.state, then handed to routes via these dependencies. Tests override the
session factory to point at an in-memory SQLite DB.
"""

from __future__ import annotations

from fastapi import Request

from server.api.config import Settings, get_settings


def get_settings_dep() -> Settings:
    return get_settings()


def get_session_factory(request: Request):
    return request.app.state.session_factory


def get_service(request: Request):
    return request.app.state.service
