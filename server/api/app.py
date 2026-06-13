"""FastAPI application factory.

create_app() wires the DB engine, session factory, evaluation service, tracelog
middleware, and routes. Production calls create_app() with no args (reads env).
Tests call it with an injected session_factory pointing at in-memory SQLite.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import sessionmaker

from server.api.config import Settings, get_settings
from server.api.hardening import RateLimitMiddleware, SecurityHeadersMiddleware
from server.api.routes import router
from server.api.service import EvaluationService
from server.api.tracelog import TracelogMiddleware
from server.db.base import Base, make_engine


def create_app(session_factory: sessionmaker | None = None,
               settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="ImmunoSense API",
        version="0.1.0",
        description="Phase 1 wellness API — log, evaluate, report.",
    )

    # Middleware order: add inner-to-outer. Starlette runs the LAST-added first,
    # so the final order per request is: RateLimit -> SecurityHeaders -> CORS ->
    # Tracelog -> route handler.
    app.add_middleware(TracelogMiddleware)
    origins = settings.cors_origin_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,          # strict allowlist (never "*")
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Dev-User", "X-Trace-Id"],
            max_age=600,
        )
    app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.enable_hsts)
    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        heavy_requests=settings.rate_limit_heavy_requests,
    )

    # DB wiring
    if session_factory is None:
        # Prefer the restricted runtime role (RLS-enforced) when configured;
        # fall back to the migration/superuser URL for dev/tests.
        runtime_url = settings.app_database_url or settings.database_url
        engine = make_engine(runtime_url)
        # In dev/SQLite, create tables if absent (prod uses Alembic migrations).
        if runtime_url.startswith("sqlite"):
            Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.service = EvaluationService(session_factory, settings)

    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "immunosense-api", "version": "0.1.0"}

    return app


# Module-level app for `uvicorn server.api.app:app`
app = create_app()
