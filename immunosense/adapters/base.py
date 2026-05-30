"""Agent adapters — the thin translation layer between the Conductor and agents.

The Sprint 5 audit established that every agent already returns the SAME
``AgentOutput`` (defined in ``immunosense.agents.base``) and already self-reports
``confidence``. So adapters do NOT normalize outputs and do NOT redefine
AgentOutput. Their job is narrow and uniform:

    1. TRANSLATE a UserBucket's AgentData into the agent's specific
       process() ``input_data`` shape (each agent wants a different dict).
    2. ISOLATE errors — wrap process() so a failure becomes a zero-confidence
       result plus an AGENT_ERROR event, never a crash that takes down the
       whole bucket evaluation (Decision 4: degrade gracefully).
    3. PROPAGATE a consistent trace_id through the call.

Adapters are thin BY DESIGN. They never build domain objects — under Option B
the caller does that using each agent's own Layer 2 pipeline. Adapters never
run ML. They are pure plumbing.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.events.bucket import AgentData
from immunosense.events.types import new_trace_id


@dataclass
class AdapterResult:
    """The outcome of running one agent through its adapter for a bucket.

    Wraps the agent's AgentOutput with adapter-level metadata: whether the
    call succeeded, how long it took, and any error string. The Conductor
    consumes this to build events and feed the quality scorer.

    Fields:
        agent_id: Which agent produced this.
        output: The agent's AgentOutput on success, or a zero/degraded
            output on failure (never None — see AgentAdapter.degraded_output).
        ok: True if process() ran without raising.
        error: Error string if ok is False, else None.
        latency_ms: Wall-clock duration of the process() call.
        trace_id: Trace id threaded through this run.
    """

    agent_id: str
    output: AgentOutput
    ok: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    trace_id: str = ""


@runtime_checkable
class AgentAdapter(Protocol):
    """Uniform contract every concrete agent adapter implements.

    Concrete adapters subclass BaseAgentAdapter (below) which provides the
    error-isolation machinery; they only implement build_input_data().
    """

    agent_id: str

    def run(self, agent_data: AgentData, bucket_end: datetime, trace_id: str) -> AdapterResult:
        ...

    def build_input_data(self, agent_data: AgentData) -> dict:
        ...


class BaseAgentAdapter:
    """Shared adapter machinery: error isolation, timing, degraded outputs.

    Concrete adapters set ``agent_id`` and implement ``build_input_data``.
    Everything else (the error-isolated run loop, the degraded fallback
    output) lives here so all five adapters behave identically on failure.
    """

    agent_id: str = "base"

    def __init__(self, agent: BaseAgent):
        """Wrap a live, initialized agent instance.

        The agent must already have had ``initialize(...)`` called if it
        needs config/mem0/trace. The adapter does not initialize agents.
        """
        if agent.agent_id != self.agent_id:
            raise ValueError(
                f"{type(self).__name__} expects agent_id={self.agent_id!r} "
                f"but got {agent.agent_id!r}"
            )
        self.agent = agent

    # ------------------------------------------------------------------ #
    # Subclasses implement this — the ONLY agent-specific logic.
    # ------------------------------------------------------------------ #
    def build_input_data(self, agent_data: AgentData) -> dict:
        """Shape an AgentData into this agent's process() input_data dict.

        Each concrete adapter overrides this to match its agent's contract
        (biomarker wants {demographics, reading}; symptoms wants
        {daily_summary}; etc.). Raising here is fine — run() isolates it.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Error-isolated execution — identical for all agents.
    # ------------------------------------------------------------------ #
    def run(
        self,
        agent_data: AgentData,
        bucket_end: datetime,
        trace_id: Optional[str] = None,
    ) -> AdapterResult:
        """Translate, call process(), isolate any failure.

        Never raises. On any exception (including a malformed input_data),
        returns an AdapterResult with ok=False and a degraded zero-confidence
        output so the Conductor can continue evaluating other agents.
        """
        trace_id = trace_id or new_trace_id(self.agent_id)
        start = time.perf_counter()
        try:
            input_data = self.build_input_data(agent_data)
            output = self.agent.process(input_data)
            # Stamp the shared trace id so Layer A can correlate.
            output.trace_id = trace_id
            latency = (time.perf_counter() - start) * 1000.0
            return AdapterResult(
                agent_id=self.agent_id,
                output=output,
                ok=True,
                error=None,
                latency_ms=latency,
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001 — deliberate broad isolation
            latency = (time.perf_counter() - start) * 1000.0
            err = f"{type(exc).__name__}: {exc}"
            return AdapterResult(
                agent_id=self.agent_id,
                output=self.degraded_output(trace_id, err),
                ok=False,
                error=err,
                latency_ms=latency,
                trace_id=trace_id,
            )

    # ------------------------------------------------------------------ #
    # Degraded fallback — a valid, zero-confidence AgentOutput.
    # ------------------------------------------------------------------ #
    def degraded_output(self, trace_id: str, error: str) -> AgentOutput:
        """Build a zero-confidence AgentOutput representing 'agent failed'.

        The vector is the agent's zero embedding, confidence is 0.0, and the
        error is recorded in both data and alerts. Quality scoring naturally
        suppresses a zero-confidence contribution, and an AGENT_ERROR event
        captures the detail for the audit trail.
        """
        dim = getattr(self.agent, "output_dim", 0)
        zero = np.zeros(dim, dtype=np.float64) if dim else np.zeros(0)
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"error": error},
            vector=zero,
            vector_dim=dim,
            alerts=[{"level": "ERROR", "message": error}],
            confidence=0.0,
            trace_id=trace_id,
        )

    @staticmethod
    def format_exc() -> str:
        """Full traceback string, for verbose debugging if a caller wants it."""
        return traceback.format_exc()
