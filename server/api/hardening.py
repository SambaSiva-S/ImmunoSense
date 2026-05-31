"""API hardening — security headers + lightweight rate limiting.

Two middlewares:
  - SecurityHeadersMiddleware: adds defensive HTTP headers to every response
    (CSP, X-Content-Type-Options, X-Frame-Options, Referrer-Policy, and HSTS in
    prod). CSP is the primary defense against XSS for the browser client.
  - RateLimitMiddleware: a simple in-memory sliding-window limiter keyed by
    client IP + authenticated user, with a tighter limit on expensive/sensitive
    endpoints (evaluate, flare, consent). Returns 429 when exceeded.

CORS is wired separately in app.py via Starlette's CORSMiddleware against the
configured origin allowlist.

NOTE (Phase 1 limitation): the rate limiter is in-memory, so limits are
per-process. For multi-worker/multi-instance deployment, move the counters to a
shared store (e.g. Redis). Documented in server/SECURITY.md. The headers
middleware has no such limitation.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Endpoints that get the tighter "heavy" limit (expensive or security-sensitive).
_HEAVY_PATHS = ("/v1/evaluate", "/v1/log/flare", "/v1/me/consent")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enable_hsts: bool = True):
        super().__init__(app)
        self.enable_hsts = enable_hsts

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        h = response.headers
        # API returns JSON; a strict CSP that forbids any active content is safe
        # and blocks injected scripts from running if a response is ever rendered.
        h.setdefault("Content-Security-Policy",
                     "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        if self.enable_hsts:
            h.setdefault("Strict-Transport-Security",
                         "max-age=31536000; includeSubDomains")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter (per IP + user)."""

    def __init__(self, app, requests: int = 120, window_seconds: int = 60,
                 heavy_requests: int = 20):
        super().__init__(app)
        self.requests = requests
        self.window = window_seconds
        self.heavy_requests = heavy_requests
        self._hits: dict[str, deque] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # IP + the dev/user header if present (best-effort identity for keying).
        ip = request.client.host if request.client else "unknown"
        user = request.headers.get("X-Dev-User") or ""
        # For real JWTs we don't decode here (cheap path); IP is the main key.
        return f"{ip}|{user}"

    async def dispatch(self, request: Request, call_next):
        if self.requests <= 0:  # disabled
            return await call_next(request)
        # Health check is never limited.
        if request.url.path == "/health":
            return await call_next(request)

        path = request.url.path
        limit = self.heavy_requests if any(path.startswith(p) for p in _HEAVY_PATHS) else self.requests
        key = f"{self._client_key(request)}|{'heavy' if limit == self.heavy_requests else 'std'}"

        now = time.monotonic()
        bucket = self._hits[key]
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            retry = int(self.window - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited",
                         "detail": "Too many requests. Please slow down.",
                         "retry_after_seconds": retry},
                headers={"Retry-After": str(retry)},
            )
        bucket.append(now)
        return await call_next(request)
