"""Tests for the API hardening pass: security headers, CORS, rate limiting."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from server.db.base import Base
from server.db import models  # noqa: F401
from server.api.app import create_app
from server.api.config import Settings


def _client(**settings_kwargs):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    settings = Settings(dev_auth=True, **settings_kwargs)
    return TestClient(create_app(session_factory=sf, settings=settings))


class TestSecurityHeaders:
    def test_headers_present_on_every_response(self):
        c = _client()
        r = c.get("/health")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in r.headers
        assert r.headers["Referrer-Policy"] == "no-referrer"

    def test_hsts_toggle(self):
        on = _client(enable_hsts=True).get("/health")
        assert "Strict-Transport-Security" in on.headers
        off = _client(enable_hsts=False).get("/health")
        assert "Strict-Transport-Security" not in off.headers


class TestCORS:
    def test_allowed_origin_echoed(self):
        c = _client(cors_origins="https://app.immunosense.com")
        r = c.get("/health", headers={"Origin": "https://app.immunosense.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.immunosense.com"

    def test_disallowed_origin_not_echoed(self):
        c = _client(cors_origins="https://app.immunosense.com")
        r = c.get("/health", headers={"Origin": "https://evil.example.com"})
        # Starlette simply omits the allow-origin header for non-allowed origins
        assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"

    def test_no_origins_configured_means_no_cors_header(self):
        c = _client(cors_origins="")
        r = c.get("/health", headers={"Origin": "https://anything.com"})
        assert r.headers.get("access-control-allow-origin") in (None,)


class TestRateLimiting:
    def test_heavy_endpoint_limited(self):
        # heavy limit set low; exceed it and expect a 429
        c = _client(rate_limit_heavy_requests=3, rate_limit_requests=1000)
        H = {"X-Dev-User": "u_rl"}
        statuses = [c.post("/v1/evaluate", headers=H).status_code for _ in range(6)]
        assert 429 in statuses, f"expected a 429 after limit, got {statuses}"

    def test_health_never_limited(self):
        c = _client(rate_limit_requests=1, rate_limit_heavy_requests=1)
        # many health calls, never limited
        assert all(c.get("/health").status_code == 200 for _ in range(10))

    def test_standard_limit_allows_normal_use(self):
        c = _client(rate_limit_requests=100, rate_limit_heavy_requests=100)
        H = {"X-Dev-User": "u_rl2"}
        # a handful of logs well under the limit
        assert all(c.post("/v1/log/symptom", headers=H, json={"fatigue": 5}).status_code == 200
                   for _ in range(5))

    def test_limit_disabled_when_zero(self):
        c = _client(rate_limit_requests=0)
        H = {"X-Dev-User": "u_rl3"}
        assert all(c.post("/v1/log/symptom", headers=H, json={"fatigue": 5}).status_code == 200
                   for _ in range(30))
