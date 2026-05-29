"""ClaudeTFM — the v1 ThinkingMachine backed by the Anthropic API.

This is the default TFM for v1. It sends the grounded prompt built by
tfm.base.build_prompt to a Claude model and returns the explanation.

SWAPPABILITY: this class implements the same `ThinkingMachine` protocol as
MockTFM and any future LocalLLMTFM (Llama via Ollama/vLLM). Swapping models is
a one-line change in how the Conductor is constructed — nothing else changes,
because the prompt and guardrails live in tfm.base, shared by all backends.

PRIVACY NOTE (documented honestly): because this calls a hosted API, the
prompt — which includes patient signal context — leaves local infrastructure.
For HIPAA-sensitive deployment, swap in a local model backend. The abstraction
exists precisely to make that swap trivial.

FAIL-SAFE: any error (missing SDK, missing key, network failure, API error)
returns a degraded TFMResponse(ok=False) carrying the safe fallback text. The
TFM never raises into the Conductor.

TESTING: the automated suite runs MockTFM, not this class, so no API key is
needed for CI. This class is exercised manually/in integration with a key set.
"""

from __future__ import annotations

import os
from typing import Optional

from immunosense.tfm.base import (
    TFMRequest,
    TFMResponse,
    ThinkingMachine,
    build_prompt,
    fallback_explanation,
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 400


class ClaudeTFM:
    """ThinkingMachine backed by the Anthropic Messages API."""

    name = "claude-tfm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 30.0,
    ):
        """Configure the Claude TFM.

        Args:
            model: Anthropic model string.
            api_key: API key; falls back to ANTHROPIC_API_KEY env var.
            max_tokens: cap on explanation length.
            timeout: per-call timeout in seconds.

        The SDK client is created lazily on first use so importing this module
        never requires the SDK or a key (keeps CI clean).
        """
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily construct the Anthropic client; raise if unavailable."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("No Anthropic API key (set ANTHROPIC_API_KEY).")
        try:
            import anthropic  # imported lazily on purpose
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed (pip install anthropic)"
            ) from e
        self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self.timeout)
        return self._client

    def explain(self, request: TFMRequest) -> TFMResponse:
        """Call Claude to explain the evaluation; degrade safely on any error."""
        trace_id = getattr(request, "trace_id", "") or ""
        system, user = build_prompt(request)
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", "") == "text"
            ).strip()
            if not text:
                raise RuntimeError("empty completion from model")
            usage = getattr(msg, "usage", None)
            return TFMResponse(
                explanation=text,
                ok=True,
                model=self.model,
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001 — fail-safe by design
            return TFMResponse(
                explanation=fallback_explanation(request),
                ok=False,
                model="fallback",
                error=f"{type(exc).__name__}: {exc}",
                trace_id=trace_id,
            )
