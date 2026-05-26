"""Base contracts every agent implements.

Adapted from Agent 4's BaseAgent. This is the universal interface the Conductor
uses to talk to all agents (via adapters).

Design principles:
    1. Every agent has identity (agent_id, agent_version) for audit trail
    2. Every agent declares its JEPA output dimension (locked in Challenge 5)
    3. Every agent emits AgentOutput with structured payload + vector + alerts
    4. Every agent reports health for ops monitoring
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class AgentOutput:
    """Universal output from any agent's process() call.

    Fields:
        agent_id: Stable identifier for the agent (e.g., "agent4_wearable")
        timestamp: When this output was generated
        data: Agent-specific structured payload (dict)
        vector: Fixed-size numerical vector for JEPA (np.ndarray)
        vector_dim: Length of the vector (declared, must match np.ndarray)
        alerts: Threshold violations or pattern matches (list of dicts)
        confidence: 0-1 data quality / output confidence
        trace_id: Unique ID for tracing this output through the system
    """

    agent_id: str
    timestamp: datetime
    data: dict
    vector: np.ndarray
    vector_dim: int
    alerts: list = field(default_factory=list)
    confidence: float = 0.0
    trace_id: str = ""

    def __post_init__(self) -> None:
        if self.vector is not None and self.vector.shape != (self.vector_dim,):
            raise ValueError(
                f"vector shape {self.vector.shape} doesn't match "
                f"declared vector_dim {self.vector_dim}"
            )


@dataclass
class AgentHealth:
    """Operational health status for one agent."""

    agent_id: str
    status: str  # "healthy" | "degraded" | "down"
    last_heartbeat: datetime
    last_success: Optional[datetime] = None
    error_count_24hr: int = 0
    avg_latency_ms: float = 0.0


class BaseAgent:
    """Universal agent interface.

    Subclasses MUST override:
        agent_id (class attribute): stable identifier
        agent_version (class attribute): semantic version string
        output_dim (class attribute): JEPA vector dimension
        process(input_data): perform inference, return AgentOutput
        get_output_vector(): return last JEPA vector (or zero vector)

    Subclasses MAY override:
        embedding_version (class attribute): defaults to ``{agent_id}_v{agent_version}``
        emit_embedding(daily_summary): JEPA-compatible embedding emission

    Standard initialization pattern::

        agent = MyAgent()
        agent.initialize(config={}, mem0_client=mem0, trace_logger=tracer)
    """

    # === Class attributes — subclasses MUST set these ===
    agent_id: str = "base"
    agent_version: str = "1.0.0"
    output_dim: int = 0
    poll_frequency: str = "1hr"  # human-readable polling cadence

    # === Class attribute — derived ===
    @property
    def embedding_version(self) -> str:
        """JEPA embedding version identifier (Challenge 5).

        Used by JEPA training to filter compatible embeddings across versions.
        Override in subclasses if embedding architecture changes independently
        of agent version.
        """
        return f"{self.agent_id}_v{self.agent_version}"

    # === Lifecycle ===
    def __init__(self) -> None:
        """Default constructor. Subclasses may override but should call super().__init__()."""
        self.config: dict = {}
        self.mem0: Optional[Any] = None
        self.trace: Optional[Any] = None
        self._error_count: int = 0
        self._last_success: Optional[datetime] = None
        self._latencies: list[float] = []

    def initialize(
        self,
        config: Optional[dict] = None,
        mem0_client: Optional[Any] = None,
        trace_logger: Optional[Any] = None,
    ) -> None:
        """Initialize agent with runtime dependencies.

        Args:
            config: Agent-specific configuration dict
            mem0_client: MemoryStore for long-term memory (Challenge 8)
            trace_logger: Trace logger for audit (Challenge 1 Layer A)
        """
        self.config = config or {}
        self.mem0 = mem0_client
        self.trace = trace_logger

    # === Core interface (must override) ===
    def process(self, input_data: dict) -> AgentOutput:
        """Process one input package and return an AgentOutput.

        Subclasses implement this. The implementation should:
            1. Validate input_data
            2. Run Layer 1/2/3 logic
            3. Build the output vector
            4. Track latency and errors
            5. Return AgentOutput with proper vector_dim

        Raises:
            NotImplementedError: If subclass didn't override.
        """
        raise NotImplementedError(f"{type(self).__name__}.process() not implemented")

    def get_output_vector(self) -> np.ndarray:
        """Return the most recent JEPA-compatible vector.

        Returns a zero/NaN vector of length output_dim if no processing has occurred.
        """
        return np.full(self.output_dim, np.nan, dtype=np.float64)

    # === JEPA compatibility (Challenge 5) ===
    def emit_embedding(self, daily_summary: Any) -> np.ndarray:
        """Emit a JEPA-compatible embedding for a daily summary.

        Default implementation returns the last output_vector. Agents with
        more sophisticated JEPA emission (e.g., Agent 5) should override.
        """
        return self.get_output_vector()

    def embedding_zero(self) -> np.ndarray:
        """Return the zero/null embedding for this agent."""
        return np.zeros(self.output_dim, dtype=np.float64)

    # === Health and observability ===
    def get_status(self) -> AgentHealth:
        """Report current health status for ops monitoring."""
        avg_lat = float(np.mean(self._latencies[-100:])) if self._latencies else 0.0
        return AgentHealth(
            agent_id=self.agent_id,
            status=self._compute_status(),
            last_heartbeat=datetime.now(timezone.utc),
            last_success=self._last_success,
            error_count_24hr=self._error_count,
            avg_latency_ms=avg_lat,
        )

    def _compute_status(self) -> str:
        """Derive health status from error counts."""
        if self._error_count > 10:
            return "down"
        if self._error_count > 3:
            return "degraded"
        return "healthy"

    def _new_trace_id(self) -> str:
        """Generate a unique trace ID for one inference call."""
        return f"{self.agent_id}-{uuid.uuid4().hex[:8]}"

    def _record_latency(self, ms: float) -> None:
        """Record one inference latency for averaging."""
        self._latencies.append(ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-500:]
