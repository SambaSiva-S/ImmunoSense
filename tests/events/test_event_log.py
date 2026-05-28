"""Tests for the NDJSON EventLog."""

import tempfile
from datetime import datetime, timezone

import pytest

from immunosense.events.event_log import EventLog
from immunosense.events.types import Event, EventType


@pytest.fixture
def log(tmp_path):
    return EventLog(tmp_path)


def _event(patient="p1", bucket="p1_2026-05-27_T2", etype=EventType.AGENT_OUTPUT,
           payload=None, agent_id=None, quality=0.5):
    return Event.create(
        patient_id=patient,
        bucket_id=bucket,
        event_type=etype,
        payload=payload or {},
        agent_id=agent_id,
        quality=quality,
    )


class TestEventLog:
    def test_append_and_read_day(self, log):
        log.append(_event())
        log.append(_event(etype=EventType.BUCKET_EVAL))
        events = log.read_day("p1", "2026-05-27")
        assert len(events) == 2

    def test_read_missing_day_empty(self, log):
        assert log.read_day("nobody", "2026-01-01") == []

    def test_append_many_groups_by_file(self, log):
        evs = [_event() for _ in range(5)]
        n = log.append_many(evs)
        assert n == 5
        assert len(log.read_day("p1", "2026-05-27")) == 5

    def test_read_bucket_filters(self, log):
        log.append(_event(bucket="p1_2026-05-27_T2"))
        log.append(_event(bucket="p1_2026-05-27_T3"))
        t2 = log.read_bucket("p1", "p1_2026-05-27_T2")
        assert len(t2) == 1
        assert t2[0].bucket_id == "p1_2026-05-27_T2"

    def test_read_range(self, log):
        log.append(_event(bucket="p1_2026-05-26_T0"))
        log.append(_event(bucket="p1_2026-05-27_T2"))
        log.append(_event(bucket="p1_2026-05-28_T1"))
        rng = log.read_range("p1", "2026-05-26", "2026-05-28")
        assert len(rng) == 3
        # sorted by timestamp
        ts = [e.timestamp for e in rng]
        assert ts == sorted(ts)

    def test_read_range_reversed_raises(self, log):
        with pytest.raises(ValueError):
            log.read_range("p1", "2026-05-28", "2026-05-26")

    def test_iter_events_filter_by_type(self, log):
        log.append(_event(etype=EventType.AGENT_OUTPUT))
        log.append(_event(etype=EventType.FLARE_BUTTON, payload={"severity": 0.9}))
        flares = list(log.iter_events("p1", EventType.FLARE_BUTTON))
        assert len(flares) == 1
        assert flares[0].payload["severity"] == 0.9

    def test_iter_events_empty_patient(self, log):
        assert list(log.iter_events("ghost")) == []

    def test_count(self, log):
        for _ in range(3):
            log.append(_event())
        assert log.count("p1") == 3

    def test_patients_listing(self, log):
        log.append(_event(patient="p1"))
        log.append(_event(patient="p2"))
        assert log.patients() == ["p1", "p2"]

    def test_persistence_across_instances(self, tmp_path):
        log1 = EventLog(tmp_path)
        log1.append(_event())
        # New instance, same root -> sees the data.
        log2 = EventLog(tmp_path)
        assert log2.count("p1") == 1

    def test_event_lands_in_bucket_date_file(self, log):
        # An event whose timestamp is just after midnight but whose bucket_id
        # says the prior day should file under the bucket's date.
        ev = Event.create(
            patient_id="p1",
            bucket_id="p1_2026-05-27_T3",
            event_type=EventType.AGENT_OUTPUT,
            payload={},
            timestamp=datetime(2026, 5, 28, 0, 0, 1, tzinfo=timezone.utc),
        )
        log.append(ev)
        # Should be readable under the bucket's embedded date.
        assert len(log.read_day("p1", "2026-05-27")) == 1
