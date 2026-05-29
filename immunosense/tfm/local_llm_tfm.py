"""LocalLLMTFM — placeholder for a local open-source model backend.

STATUS: scaffold only (not the v1 default; ClaudeTFM is v1).

This is the documented swap target for running the TFM on a LOCAL open-source
model (e.g. Llama 3.1 8B or 3.3 70B) served via Ollama or vLLM. Filling this in
is the path to keeping patient context on local infrastructure (the HIPAA-
friendly option) without changing anything else in the system — it implements
the same `ThinkingMachine` protocol and reuses the shared grounded prompt from
tfm.base.

To implement against Ollama (default endpoint http://localhost:11434):
    - POST to /api/chat with {"model": "llama3.1:8b", "messages": [...],
      "stream": false}
    - map the shared (system, user) prompt into the messages array
    - parse message.content from the response
    - keep the same fail-safe behavior: return TFMResponse(ok=False) with the
      fallback explanation on any error, never raise.

To implement against vLLM: point an OpenAI-compatible client at the vLLM
server's /v1/chat/completions endpoint; the prompt construction is identical.
"""

from __future__ import annotations

from immunosense.tfm.base import (
    TFMRequest,
    TFMResponse,
    build_prompt,
    fallback_explanation,
)

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_LOCAL_MODEL = "llama3.1:8b"


class LocalLLMTFM:
    """Local open-source model TFM backend (Ollama/vLLM). Scaffold for v1.1.

    Constructed and protocol-conformant now so the swap is wiring, not a
    rewrite. The explain() body is intentionally left as a clear NotImplemented
    so it is obvious this is not yet active.
    """

    name = "local-llm-tfm"

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        timeout: float = 60.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def explain(self, request: TFMRequest) -> TFMResponse:
        # Not yet implemented. Until then, behave safely: return the
        # deterministic fallback rather than raising, so accidentally wiring
        # this backend cannot crash an evaluation.
        return TFMResponse(
            explanation=fallback_explanation(request),
            ok=False,
            model="fallback",
            error="LocalLLMTFM not implemented yet (scaffold for local-model swap)",
            trace_id=getattr(request, "trace_id", "") or "",
        )
