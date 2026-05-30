"""Tracelog + error tracking middleware.

Every request gets a trace_id (propagated to the Conductor's events so a bucket
evaluation can be tied back to the HTTP request that triggered it). Requests and
errors are logged as structured records. Unhandled exceptions are captured,
logged with their trace_id, and returned as a clean JSON error (never a leaked
stack trace to the client).

This is the "track any breaks or errors" mechanism: grep the logs by trace_id to
follow one request end-to-end, including which agent failed if one did.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from contextvars import ContextVar

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# The current request's trace id, readable anywhere downstream (e.g. when
# building the Conductor so its events carry the same trace id).
current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")

logger = logging.getLogger("immunosense.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [trace=%(trace_id)s] %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class _TraceAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["trace_id"] = current_trace_id.get() or "-"
        return msg, kwargs


log = _TraceAdapter(logger, {})


class TracelogMiddleware(BaseHTTPMiddleware):
    """Assigns a trace id, logs request/response, captures unhandled errors."""

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
        token = current_trace_id.set(trace_id)
        start = time.monotonic()
        method, path = request.method, request.url.path
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 — we capture everything here
            elapsed = (time.monotonic() - start) * 1000
            log.error(
                f"{method} {path} -> UNHANDLED {type(exc).__name__}: {exc} "
                f"({elapsed:.0f}ms)\n{traceback.format_exc()}"
            )
            current_trace_id.reset(token)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "detail": "An unexpected error occurred.",
                    "trace_id": trace_id,
                },
                headers={"X-Trace-Id": trace_id},
            )
        elapsed = (time.monotonic() - start) * 1000
        log.info(f"{method} {path} -> {response.status_code} ({elapsed:.0f}ms)")
        response.headers["X-Trace-Id"] = trace_id
        current_trace_id.reset(token)
        return response


def get_trace_id() -> str:
    """The current request's trace id (empty string outside a request)."""
    return current_trace_id.get()
