"""Tests for PostgresEventLog — drop-in parity with the NDJSON EventLog,
audit logging, and a real Conductor integration run."""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.adapters import AdapterRegistry, SymptomsMoodAdapter, WearableAdapter
from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.conductor import Conductor
from immunosense.events import AgentData, BucketBuilder, EventType, UserBucket
from immunosense.events.types import Event
from server.db.event_store import PostgresEventLog
from server.db.models import AccessLog


def _event(user="u1", bucket="u1_2026-05-27_T2", etype=EventType.AGENT_OUTPUT,
           payload=None, agent_id=None, quality=0.5, ts=None):
    return Event.create(
        user_id=user, bucket_id=bucket, event_type=etype,
        payload=payload or {}, agent_id=agent_id, quality=quality,
        timestamp=ts or datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc),
    )


class TestEventStoreBasics:
    def test_append_and_read_bucket(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event())
        events = log.read_bucket("u1", "u1_2026-05-27_T2")
        assert len(events) == 1
        assert events[0].user_id == "u1"

    def test_append_many(self, session_factory):
        log = PostgresEventLog(session_factory)
        n = log.append_many([_event(), _event(etype=EventType.BUCKET_EVAL)])
        assert n == 2
        assert log.count("u1") == 2

    def test_read_day(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event())
        assert len(log.read_day("u1", "2026-05-27")) == 1
        assert log.read_day("u1", "2026-01-01") == []

    def test_read_range(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event(bucket="u1_2026-05-26_T0",
                          ts=datetime(2026, 5, 26, 2, 0, tzinfo=timezone.utc)))
        log.append(_event(bucket="u1_2026-05-27_T2"))
        log.append(_event(bucket="u1_2026-05-28_T1",
                          ts=datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc)))
        rng = log.read_range("u1", "2026-05-26", "2026-05-28")
        assert len(rng) == 3
        ts = [e.timestamp for e in rng]
        assert ts == sorted(ts)

    def test_read_range_reversed_raises(self, session_factory):
        log = PostgresEventLog(session_factory)
        with pytest.raises(ValueError):
            log.read_range("u1", "2026-05-28", "2026-05-26")

    def test_iter_events_filter(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event(etype=EventType.AGENT_OUTPUT))
        log.append(_event(etype=EventType.FLARE_BUTTON, payload={"severity": 0.9}))
        flares = list(log.iter_events("u1", EventType.FLARE_BUTTON))
        assert len(flares) == 1
        assert flares[0].payload["severity"] == 0.9

    def test_count_and_users(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event(user="u1"))
        log.append(_event(user="u2"))
        assert log.count("u1") == 1
        assert log.users() == ["u1", "u2"]
        assert log.patients() == ["u1", "u2"]  # back-compat alias

    def test_event_round_trip_fidelity(self, session_factory):
        log = PostgresEventLog(session_factory)
        ev = _event(payload={"vector": [0.1, 0.2], "alerts": []},
                    agent_id="agent5_symptoms_mood", quality=0.82)
        log.append(ev)
        got = log.read_bucket("u1", "u1_2026-05-27_T2")[0]
        assert got.event_id == ev.event_id
        assert got.agent_id == "agent5_symptoms_mood"
        assert got.quality == 0.82
        assert got.payload == {"vector": [0.1, 0.2], "alerts": []}
        assert got.event_type == EventType.AGENT_OUTPUT


class TestAuditLogging:
    def test_append_writes_audit_row(self, session_factory):
        log = PostgresEventLog(session_factory)
        log.append(_event())
        with session_factory() as s:
            from sqlalchemy import select, func
            n = s.execute(select(func.count()).select_from(AccessLog)).scalar_one()
        assert n == 1

    def test_audit_can_be_disabled(self, session_factory):
        log = PostgresEventLog(session_factory, audit=False)
        log.append(_event())
        with session_factory() as s:
            from sqlalchemy import select, func
            n = s.execute(select(func.count()).select_from(AccessLog)).scalar_one()
        assert n == 0


class _Elevated(BaseAgent):
    def __init__(self, aid, dim, poll):
        super().__init__()
        self.agent_id = aid
        self.output_dim = dim
        self.poll_frequency = poll

    def process(self, d):
        return AgentOutput(
            agent_id=self.agent_id, timestamp=datetime.now(timezone.utc),
            data={"ok": True}, vector=np.ones(self.output_dim) * 0.8,
            vector_dim=self.output_dim, alerts=[{"severity": "critical"}],
            confidence=0.9,
        )


class TestConductorIntegration:
    """The real Conductor must run against PostgresEventLog unchanged."""

    def test_full_evaluation_persists(self, session_factory):
        log = PostgresEventLog(session_factory)
        registry = AdapterRegistry()
        registry.register(SymptomsMoodAdapter(_Elevated("agent5_symptoms_mood", 36, "daily")))
        registry.register(WearableAdapter(_Elevated("agent4_wearable", 29, "1hr")))
        conductor = Conductor(registry=registry, event_log=log, disease="SLE")

        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        bucket = BucketBuilder.bucket_for("u_int_1", ts)
        ub = UserBucket(bucket=bucket)
        ub.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        ub.add(AgentData("agent4_wearable",
                         {"night_df": "df", "rr_intervals": [800], "night_idx": 1},
                         produced_at=ts))
        report = conductor.evaluate_bucket(ub)

        # Inference produced
        assert report.flare_probability is not None
        assert "autonomic_stress" in [p.name for p in report.matched_patterns]

        # Persisted to the DB
        events = log.read_bucket("u_int_1", bucket.bucket_id)
        types = sorted(e.event_type.value for e in events)
        assert types.count("agent_output") == 2
        assert "bucket_eval" in types

        # Single shared trace id across all events
        assert len({e.trace_id for e in events}) == 1

    def test_flare_button_persists(self, session_factory):
        log = PostgresEventLog(session_factory)
        registry = AdapterRegistry()
        registry.register(SymptomsMoodAdapter(_Elevated("agent5_symptoms_mood", 36, "daily")))
        conductor = Conductor(registry=registry, event_log=log, disease="SLE")
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        bucket = BucketBuilder.bucket_for("u_int_2", ts)
        ub = UserBucket(bucket=bucket)
        ub.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        conductor.on_flare_button(ub, severity=0.9)
        flares = [e for e in log.read_bucket("u_int_2", bucket.bucket_id)
                  if e.event_type == EventType.FLARE_BUTTON]
        assert len(flares) == 1
        assert flares[0].payload["severity"] == 0.9
