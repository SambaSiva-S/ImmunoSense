"""Shared fixtures for adapter + conductor tests.

Uses lightweight fake agents that subclass BaseAgent so tests are fast and
deterministic (no trained ML models, no API calls). The fakes honor the same
contract the real agents do: process(input_data) -> AgentOutput.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput, BaseAgent


class FakeAgent(BaseAgent):
    """Configurable fake agent for testing adapters and the Conductor."""

    def __init__(self, agent_id, output_dim, poll_frequency, confidence=0.9,
                 required_key="payload", should_fail=False):
        super().__init__()
        self.agent_id = agent_id
        self.output_dim = output_dim
        self.poll_frequency = poll_frequency
        self._confidence = confidence
        self._required_key = required_key
        self._should_fail = should_fail

    def process(self, input_data: dict) -> AgentOutput:
        if self._should_fail:
            raise RuntimeError("intentional test failure")
        if self._required_key not in input_data:
            raise ValueError(f"missing required key {self._required_key!r}")
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"echo": input_data.get(self._required_key)},
            vector=np.ones(self.output_dim, dtype=np.float64) * 0.5,
            vector_dim=self.output_dim,
            alerts=[],
            confidence=self._confidence,
        )


@pytest.fixture
def fake_biomarker():
    # Mimics biomarker contract: needs 'demographics' + 'reading'.
    class FakeBiomarker(FakeAgent):
        def process(self, input_data):
            if "demographics" not in input_data or "reading" not in input_data:
                raise ValueError("needs demographics and reading")
            return AgentOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                data={"ok": True},
                vector=np.ones(self.output_dim) * 0.5,
                vector_dim=self.output_dim,
                confidence=self._confidence,
            )
    return FakeBiomarker("agent1_biomarker", 7, "weekly", confidence=0.9)


@pytest.fixture
def fake_symptoms():
    class FakeSymptoms(FakeAgent):
        def process(self, input_data):
            if "daily_summary" not in input_data:
                raise ValueError("needs daily_summary")
            return AgentOutput(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                data={"ok": True},
                vector=np.ones(self.output_dim) * 0.3,
                vector_dim=self.output_dim,
                confidence=self._confidence,
            )
    return FakeSymptoms("agent5_symptoms_mood", 36, "daily", confidence=0.8)
