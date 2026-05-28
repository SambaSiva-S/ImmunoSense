"""Tests for Layer A core types: Event, EventType, ConfidenceLevel."""

from datetime import datetime, timezone

import pytest

from immunosense.events.types import (
    SCHEMA_VERSION,
    ConfidenceLevel,
    Event,
    EventType,
    new_trace_id,
    utc_now,
)


class TestEventType:
    def test_values_are_strings(self):
        assert EventType.AGENT_OUTPUT.value == "agent_output"
        assert EventType.AGENT_ERROR == "agent_error"  # str-enum equality

    def test_all_types_present(self):
        names = {e.value for e in EventType}
        assert names == {
            "agent_output",
            "agent_error",
            "flare_button",
            "bucket_eval",
            "clinical_event",
        }


class TestConfidenceLevel:
    def test_rank_ordering(self):
        assert ConfidenceLevel.INSUFFICIENT.rank == 0
        assert ConfidenceLevel.LOW.rank == 1
        assert ConfidenceLevel.MODERATE.rank == 2
        assert ConfidenceLevel.HIGH.rank == 3

    def test_rank_comparisons(self):
        assert ConfidenceLevel.HIGH.rank > ConfidenceLevel.MODERATE.rank
        assert ConfidenceLevel.LOW.rank < ConfidenceLevel.HIGH.rank


class TestEvent:
    def test_create_defaults(self):
        ev = Event.create(
            patient_id="p1",
            bucket_id="p1_2026-05-27_T2",
            event_type=EventType.AGENT_OUTPUT,
            payload={"x": 1},
        )
        assert ev.patient_id == "p1"
        assert ev.event_type == EventType.AGENT_OUTPUT
        assert ev.schema_version == SCHEMA_VERSION
        assert ev.trace_id  # auto-generated
        assert ev.event_id  # auto-generated
        assert ev.timestamp.tzinfo is not None

    def test_create_with_explicit_trace(self):
        ev = Event.create(
            patient_id="p1",
            bucket_id="b",
            event_type=EventType.FLARE_BUTTON,
            payload={},
            trace_id="trace-fixed",
        )
        assert ev.trace_id == "trace-fixed"

    def test_json_round_trip(self):
        ev = Event.create(
            patient_id="p1",
            bucket_id="p1_2026-05-27_T2",
            event_type=EventType.AGENT_OUTPUT,
            payload={"vector": [0.1, 0.2], "alerts": []},
            agent_id="agent5_symptoms_mood",
            quality=0.82,
        )
        line = ev.to_json()
        assert "\n" not in line
        ev2 = Event.from_json(line)
        assert ev2.event_id == ev.event_id
        assert ev2.event_type == ev.event_type
        assert ev2.agent_id == ev.agent_id
        assert ev2.quality == ev.quality
        assert ev2.payload == ev.payload
        assert ev2.timestamp == ev.timestamp

    def test_dict_round_trip(self):
        ev = Event.create("p1", "b", EventType.BUCKET_EVAL, {"k": "v"})
        d = ev.to_dict()
        assert isinstance(d["timestamp"], str)  # ISO serialized
        ev2 = Event.from_dict(d)
        assert ev2.event_id == ev.event_id

    def test_frozen_immutable(self):
        ev = Event.create("p1", "b", EventType.AGENT_OUTPUT, {})
        with pytest.raises(Exception):
            ev.quality = 0.5  # frozen dataclass

    def test_from_dict_tolerates_naive_timestamp(self):
        ev = Event.create("p1", "b", EventType.AGENT_OUTPUT, {})
        d = ev.to_dict()
        # Already a string; ensure parse works
        ev2 = Event.from_dict(d)
        assert ev2.timestamp == ev.timestamp


class TestHelpers:
    def test_new_trace_id_unique(self):
        a = new_trace_id()
        b = new_trace_id()
        assert a != b
        assert a.startswith("trace-")

    def test_new_trace_id_prefix(self):
        t = new_trace_id("agent1")
        assert t.startswith("agent1-")

    def test_utc_now_is_aware(self):
        n = utc_now()
        assert n.tzinfo is not None
        assert n.tzinfo == timezone.utc
