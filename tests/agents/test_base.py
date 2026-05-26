"""Tests for BaseAgent, AgentOutput, AgentHealth contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentHealth, AgentOutput, BaseAgent


# ============================================================
# AgentOutput dataclass tests
# ============================================================

class TestAgentOutput:
    def test_construct_valid(self):
        vec = np.zeros(5, dtype=np.float64)
        out = AgentOutput(
            agent_id="test_agent",
            timestamp=datetime.now(timezone.utc),
            data={"k": "v"},
            vector=vec,
            vector_dim=5,
        )
        assert out.agent_id == "test_agent"
        assert out.vector_dim == 5
        assert out.alerts == []  # default
        assert out.confidence == 0.0  # default
        assert out.trace_id == ""  # default

    def test_construct_with_all_fields(self):
        vec = np.ones(3, dtype=np.float64)
        out = AgentOutput(
            agent_id="test",
            timestamp=datetime.now(timezone.utc),
            data={},
            vector=vec,
            vector_dim=3,
            alerts=[{"name": "alert1"}],
            confidence=0.85,
            trace_id="trace-abc123",
        )
        assert out.alerts == [{"name": "alert1"}]
        assert out.confidence == 0.85
        assert out.trace_id == "trace-abc123"

    def test_vector_dim_mismatch_raises(self):
        vec = np.zeros(5, dtype=np.float64)
        with pytest.raises(ValueError, match="vector shape"):
            AgentOutput(
                agent_id="test",
                timestamp=datetime.now(timezone.utc),
                data={},
                vector=vec,
                vector_dim=10,  # mismatch
            )


# ============================================================
# AgentHealth dataclass tests
# ============================================================

class TestAgentHealth:
    def test_construct_healthy(self):
        h = AgentHealth(
            agent_id="test",
            status="healthy",
            last_heartbeat=datetime.now(timezone.utc),
        )
        assert h.status == "healthy"
        assert h.last_success is None
        assert h.error_count_24hr == 0


# ============================================================
# BaseAgent contract tests
# ============================================================

class DummyAgent(BaseAgent):
    """Minimal BaseAgent subclass for testing."""

    agent_id = "dummy"
    agent_version = "0.1.0"
    output_dim = 4

    def process(self, input_data: dict) -> AgentOutput:
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"input": input_data},
            vector=vec,
            vector_dim=self.output_dim,
            confidence=1.0,
            trace_id=self._new_trace_id(),
        )

    def get_output_vector(self) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0, 4.0])


class TestBaseAgent:
    def test_subclass_must_override_process(self):
        class IncompleteAgent(BaseAgent):
            agent_id = "incomplete"
            output_dim = 1

        agent = IncompleteAgent()
        with pytest.raises(NotImplementedError):
            agent.process({})

    def test_default_get_output_vector_is_nan(self):
        """Default get_output_vector returns NaN of correct dimension."""
        class A(BaseAgent):
            agent_id = "a"
            output_dim = 3

        agent = A()
        v = agent.get_output_vector()
        assert v.shape == (3,)
        assert np.isnan(v).all()

    def test_embedding_zero_is_zeros(self):
        class A(BaseAgent):
            agent_id = "a"
            output_dim = 5

        agent = A()
        z = agent.embedding_zero()
        assert z.shape == (5,)
        assert (z == 0).all()

    def test_embedding_version_default(self):
        """Default embedding_version is derived from agent_id + version."""
        class A(BaseAgent):
            agent_id = "agent_x"
            agent_version = "1.2.3"
            output_dim = 1

        agent = A()
        assert agent.embedding_version == "agent_x_v1.2.3"

    def test_initialize_sets_attributes(self):
        agent = DummyAgent()
        mock_mem0 = object()
        mock_trace = object()
        agent.initialize(
            config={"key": "value"},
            mem0_client=mock_mem0,
            trace_logger=mock_trace,
        )
        assert agent.config == {"key": "value"}
        assert agent.mem0 is mock_mem0
        assert agent.trace is mock_trace

    def test_initialize_with_defaults(self):
        agent = DummyAgent()
        agent.initialize()
        assert agent.config == {}
        assert agent.mem0 is None
        assert agent.trace is None

    def test_get_status_default_healthy(self):
        agent = DummyAgent()
        status = agent.get_status()
        assert status.agent_id == "dummy"
        assert status.status == "healthy"
        assert status.error_count_24hr == 0

    def test_get_status_degraded_after_errors(self):
        agent = DummyAgent()
        agent._error_count = 5
        status = agent.get_status()
        assert status.status == "degraded"

    def test_get_status_down_after_many_errors(self):
        agent = DummyAgent()
        agent._error_count = 15
        status = agent.get_status()
        assert status.status == "down"

    def test_trace_id_unique(self):
        agent = DummyAgent()
        ids = {agent._new_trace_id() for _ in range(100)}
        assert len(ids) == 100  # all unique

    def test_trace_id_format(self):
        agent = DummyAgent()
        tid = agent._new_trace_id()
        assert tid.startswith("dummy-")
        # 8 hex chars after the prefix
        assert len(tid) == len("dummy-") + 8

    def test_record_latency(self):
        agent = DummyAgent()
        for ms in [10, 20, 30, 40, 50]:
            agent._record_latency(ms)
        status = agent.get_status()
        assert status.avg_latency_ms == 30.0

    def test_latency_bounded(self):
        """Latency history shouldn't grow unboundedly."""
        agent = DummyAgent()
        for _ in range(2000):
            agent._record_latency(10.0)
        # Should have been truncated
        assert len(agent._latencies) < 1000


class TestDummyAgentEndToEnd:
    """Integration tests via the dummy subclass."""

    def test_process_returns_valid_output(self):
        agent = DummyAgent()
        out = agent.process({"some": "input"})
        assert isinstance(out, AgentOutput)
        assert out.agent_id == "dummy"
        assert out.vector.shape == (4,)
        assert out.confidence == 1.0
        assert out.trace_id.startswith("dummy-")
