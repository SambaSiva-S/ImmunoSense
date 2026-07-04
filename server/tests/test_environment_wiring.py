"""Environment agent wiring (Phase 2) — builder + adapter path."""
from datetime import datetime, timezone

from immunosense.adapters import EnvironmentAdapter
from immunosense.agents.environment.agent import EnvironmentAgent
from immunosense.agents.environment.sources.mock import MockEnvironmentSource
from immunosense.events.bucket import AgentData
from immunosense.events.types import new_trace_id
from server.builders.environment_builder import build_environment_summary


class TestEnvironmentBuilder:
    def test_builds_summary_from_coords(self):
        s = build_environment_summary(
            lat=35.227, lon=-80.843, target_date="2026-07-03",
            label="Charlotte, NC", source=MockEnvironmentSource(),
        )
        # all five features present, location carries the label + coords
        assert s.date == "2026-07-03"
        assert s.location["label"] == "Charlotte, NC"
        assert s.location["lat"] == 35.227
        assert s.pm25_ug_m3 is not None
        assert s.pollen_index is not None
        assert set(s.threshold_alerts) == {"pm25", "ozone", "uv", "barometric", "pollen"}

    def test_deterministic_for_same_inputs(self):
        a = build_environment_summary(35.2, -80.8, "2026-07-03", source=MockEnvironmentSource())
        b = build_environment_summary(35.2, -80.8, "2026-07-03", source=MockEnvironmentSource())
        assert a.pm25_ug_m3 == b.pm25_ug_m3  # mock is seed-deterministic

    def test_adapter_processes_without_crash(self):
        s = build_environment_summary(35.227, -80.843, "2026-07-03",
                                      source=MockEnvironmentSource())
        adapter = EnvironmentAdapter(EnvironmentAgent())
        ad = AgentData("agent3_environment", s, produced_at=datetime.now(timezone.utc))
        res = adapter.run(ad, datetime.now(timezone.utc), new_trace_id())
        assert res.ok
        assert res.agent_id == "agent3_environment"
