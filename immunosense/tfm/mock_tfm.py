"""MockTFM — a deterministic ThinkingMachine for tests and offline use.

Produces the safe, deterministic fallback explanation without any model call.
This is what the automated test suite runs against, so CI needs no API key,
no network, and no GPU. It also serves as a real fallback the Conductor can use
when a model backend is unavailable.
"""

from __future__ import annotations

from immunosense.tfm.base import (
    TFMRequest,
    TFMResponse,
    ThinkingMachine,
    fallback_explanation,
)


class MockTFM:
    """Deterministic TFM. Same input -> same explanation, always."""

    name = "mock-tfm"

    def explain(self, request: TFMRequest) -> TFMResponse:
        text = fallback_explanation(request)
        return TFMResponse(
            explanation=text,
            ok=True,
            model=self.name,
            trace_id=getattr(request, "trace_id", "") or "",
        )
