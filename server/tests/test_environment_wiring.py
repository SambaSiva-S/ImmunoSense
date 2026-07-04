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


class TestDestinationCheck:
    def test_check_returns_assessment(self, api_client, monkeypatch):
        # mock geocode so the test doesn't hit the network
        monkeypatch.setattr("server.api.geocode.geocode",
                            lambda q, **k: (35.227, -80.843, "Charlotte, NC 28202"))
        r = api_client.post("/v1/environment/check", headers={"X-Dev-User": "u_dest"},
                            json={"location": "28202"})
        assert r.status_code == 200
        body = r.json()
        assert body["location"] == "Charlotte, NC 28202"
        assert "readings" in body and "pollen" in body["readings"]
        assert "headline" in body and "disclaimer" in body

    def test_check_bad_location(self, api_client, monkeypatch):
        monkeypatch.setattr("server.api.geocode.geocode", lambda q, **k: None)
        r = api_client.post("/v1/environment/check", headers={"X-Dev-User": "u_dest2"},
                            json={"location": "zznowhere"})
        assert r.status_code == 200
        assert "error" in r.json()

    def test_check_empty_location(self, api_client):
        r = api_client.post("/v1/environment/check", headers={"X-Dev-User": "u_dest3"},
                            json={"location": ""})
        assert r.status_code == 200
        assert "error" in r.json()
