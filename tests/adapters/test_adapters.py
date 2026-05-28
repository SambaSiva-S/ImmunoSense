"""Tests for agent adapters: translation, error isolation, registry."""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.adapters import (
    AdapterRegistry,
    BiomarkerAdapter,
    DietaryAdapter,
    EnvironmentAdapter,
    SymptomsMoodAdapter,
    WearableAdapter,
)
from immunosense.adapters.base import AdapterResult, BaseAgentAdapter
from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.events.bucket import AgentData


class _FakeAgent(BaseAgent):
    def __init__(self, agent_id, output_dim, poll, fail=False, conf=0.9):
        super().__init__()
        self.agent_id = agent_id
        self.output_dim = output_dim
        self.poll_frequency = poll
        self._fail = fail
        self._conf = conf

    def process(self, input_data):
        if self._fail:
            raise RuntimeError("boom")
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"input_keys": sorted(input_data.keys())},
            vector=np.ones(self.output_dim),
            vector_dim=self.output_dim,
            confidence=self._conf,
        )


class TestBiomarkerAdapter:
    def test_dict_shape(self):
        agent = _FakeAgent("agent1_biomarker", 7, "weekly")
        adapter = BiomarkerAdapter(agent)
        data = AgentData(
            agent_id="agent1_biomarker",
            domain_object={"demographics": {"age": 40}, "reading": {"CRP": 2.0}},
        )
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok
        assert result.output.data["input_keys"] == ["demographics", "reading"]

    def test_missing_keys_isolated(self):
        agent = _FakeAgent("agent1_biomarker", 7, "weekly")
        adapter = BiomarkerAdapter(agent)
        data = AgentData(agent_id="agent1_biomarker", domain_object={"wrong": 1})
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok is False
        assert result.output.confidence == 0.0
        assert result.output.vector_dim == 7

    def test_wrong_agent_id_rejected(self):
        agent = _FakeAgent("agent5_symptoms_mood", 36, "daily")
        with pytest.raises(ValueError):
            BiomarkerAdapter(agent)


class TestErrorIsolation:
    def test_agent_raises_becomes_degraded(self):
        agent = _FakeAgent("agent1_biomarker", 7, "weekly", fail=True)
        adapter = BiomarkerAdapter(agent)
        data = AgentData(
            agent_id="agent1_biomarker",
            domain_object={"demographics": {}, "reading": {}},
        )
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok is False
        assert "boom" in result.error
        assert result.output.confidence == 0.0
        assert len(result.output.alerts) == 1
        assert result.output.alerts[0]["level"] == "ERROR"

    def test_trace_id_propagates(self):
        agent = _FakeAgent("agent5_symptoms_mood", 36, "daily")
        adapter = SymptomsMoodAdapter(agent)
        data = AgentData(agent_id="agent5_symptoms_mood", domain_object="summary")
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc),
                             trace_id="trace-xyz")
        assert result.trace_id == "trace-xyz"
        assert result.output.trace_id == "trace-xyz"

    def test_auto_trace_id_when_absent(self):
        agent = _FakeAgent("agent5_symptoms_mood", 36, "daily")
        adapter = SymptomsMoodAdapter(agent)
        data = AgentData(agent_id="agent5_symptoms_mood", domain_object="s")
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.trace_id.startswith("agent5_symptoms_mood-")

    def test_latency_recorded(self):
        agent = _FakeAgent("agent5_symptoms_mood", 36, "daily")
        adapter = SymptomsMoodAdapter(agent)
        data = AgentData(agent_id="agent5_symptoms_mood", domain_object="s")
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.latency_ms >= 0.0


class TestOtherAdapters:
    def test_dietary_rollup(self):
        agent = _FakeAgent("agent2_dietary", 10, "daily")
        adapter = DietaryAdapter(agent)
        data = AgentData(agent_id="agent2_dietary", domain_object="rollup_obj")
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok
        assert result.output.data["input_keys"] == ["rollup"]

    def test_dietary_with_flares_extra(self):
        agent = _FakeAgent("agent2_dietary", 10, "daily")
        adapter = DietaryAdapter(agent)
        data = AgentData(
            agent_id="agent2_dietary",
            domain_object="rollup_obj",
            extras={"flares": [("2026-05-27", 0.8)]},
        )
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok
        assert sorted(result.output.data["input_keys"]) == ["flares", "rollup"]

    def test_environment_summary(self):
        agent = _FakeAgent("agent3_environment", 5, "6hr")
        adapter = EnvironmentAdapter(agent)
        data = AgentData(agent_id="agent3_environment", domain_object="env_summary")
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok
        assert result.output.data["input_keys"] == ["daily_summary"]

    def test_wearable_requires_keys(self):
        agent = _FakeAgent("agent4_wearable", 29, "1hr")
        adapter = WearableAdapter(agent)
        # Missing required keys -> isolated failure.
        data = AgentData(agent_id="agent4_wearable", domain_object={"night_df": 1})
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok is False

    def test_wearable_full_input(self):
        agent = _FakeAgent("agent4_wearable", 29, "1hr")
        adapter = WearableAdapter(agent)
        data = AgentData(
            agent_id="agent4_wearable",
            domain_object={"night_df": "df", "rr_intervals": [800, 810], "night_idx": 1},
        )
        result = adapter.run(data, bucket_end=datetime.now(timezone.utc))
        assert result.ok


class TestAdapterRegistry:
    def test_register_and_get(self):
        agent = _FakeAgent("agent1_biomarker", 7, "weekly")
        reg = AdapterRegistry()
        reg.register(BiomarkerAdapter(agent))
        assert reg.has("agent1_biomarker")
        assert reg.get("agent1_biomarker") is not None
        assert reg.get("missing") is None
        assert len(reg) == 1

    def test_from_agents(self):
        agents = [
            _FakeAgent("agent1_biomarker", 7, "weekly"),
            _FakeAgent("agent5_symptoms_mood", 36, "daily"),
        ]
        reg = AdapterRegistry.from_agents(agents)
        assert reg.agent_ids == ["agent1_biomarker", "agent5_symptoms_mood"]

    def test_from_agents_unknown_id_raises(self):
        agent = _FakeAgent("agent99_unknown", 3, "daily")
        with pytest.raises(ValueError):
            AdapterRegistry.from_agents([agent])
