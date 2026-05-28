"""Tests for TimeBucket, BucketBuilder, PatientBucket, and freshness."""

from datetime import datetime, timedelta, timezone

import pytest

from immunosense.events.bucket import (
    BUCKETS_PER_DAY,
    AgentData,
    BucketBuilder,
    PatientBucket,
    TimeBucket,
    freshness_weight,
)


class TestBucketBuilder:
    def test_four_buckets_per_day(self):
        assert BUCKETS_PER_DAY == 4

    @pytest.mark.parametrize(
        "hour,expected_label",
        [(0, "T0"), (5, "T0"), (6, "T1"), (11, "T1"),
         (12, "T2"), (17, "T2"), (18, "T3"), (23, "T3")],
    )
    def test_hour_to_label(self, hour, expected_label):
        ts = datetime(2026, 5, 27, hour, 30, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("p1", ts)
        assert b.label == expected_label

    def test_bucket_id_format(self):
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("patient001", ts)
        assert b.bucket_id == "patient001_2026-05-27_T2"

    def test_window_bounds(self):
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("p1", ts)
        assert b.start == datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc)
        assert b.end == datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)

    def test_contains(self):
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("p1", ts)
        assert b.contains(ts)
        assert b.contains(b.start)
        assert not b.contains(b.end)  # end is exclusive
        assert not b.contains(b.start - timedelta(seconds=1))

    def test_naive_timestamp_treated_as_utc(self):
        ts = datetime(2026, 5, 27, 14, 30)  # naive
        b = BucketBuilder.bucket_for("p1", ts)
        assert b.label == "T2"

    def test_round_trip_simple_id(self):
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("patient001", ts)
        rt = BucketBuilder.from_bucket_id(b.bucket_id)
        assert rt.patient_id == "patient001"
        assert rt.label == "T2"
        assert rt.start == b.start
        assert rt.end == b.end

    def test_round_trip_underscore_patient(self):
        ts = datetime(2026, 5, 27, 8, 0, tzinfo=timezone.utc)
        b = BucketBuilder.bucket_for("patient_001_abc", ts)
        rt = BucketBuilder.from_bucket_id(b.bucket_id)
        assert rt.patient_id == "patient_001_abc"
        assert rt.label == "T1"

    def test_malformed_bucket_id_raises(self):
        with pytest.raises(ValueError):
            BucketBuilder.from_bucket_id("not-a-valid-id")


class TestFreshness:
    def test_none_produced_at_is_fully_fresh(self):
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        assert freshness_weight("daily", None, ref) == 1.0

    def test_zero_age_is_fully_fresh(self):
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        assert freshness_weight("1hr", ref, ref) == pytest.approx(1.0)

    def test_weekly_vs_hourly_same_age(self):
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        six_days = ref - timedelta(days=6)
        w_weekly = freshness_weight("weekly", six_days, ref)
        w_hourly = freshness_weight("1hr", six_days, ref)
        # A 6-day-old weekly reading is still fresh; hourly is stale.
        assert w_weekly > 0.5
        assert w_hourly < 0.01
        assert w_weekly > w_hourly

    def test_halflife_behavior(self):
        # At exactly one half-life, weight should be ~0.5.
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        # daily halflife is 36h
        t = ref - timedelta(hours=36)
        assert freshness_weight("daily", t, ref) == pytest.approx(0.5, abs=0.01)

    def test_unknown_cadence_uses_default(self):
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        t = ref - timedelta(hours=36)
        # unknown -> default halflife 36h -> ~0.5
        assert freshness_weight("monthly", t, ref) == pytest.approx(0.5, abs=0.01)

    def test_future_produced_at_clamped(self):
        ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
        future = ref + timedelta(hours=5)
        # negative age clamps to 0 -> fully fresh
        assert freshness_weight("daily", future, ref) == pytest.approx(1.0)


class TestPatientBucket:
    def _bucket(self):
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        return BucketBuilder.bucket_for("p1", ts)

    def test_add_and_get(self):
        pb = PatientBucket(bucket=self._bucket())
        pb.add(AgentData(agent_id="agent1_biomarker", domain_object={"x": 1}))
        assert pb.has_agent("agent1_biomarker")
        assert pb.get("agent1_biomarker").domain_object == {"x": 1}
        assert pb.get("missing") is None

    def test_reporting_agents_sorted(self):
        pb = PatientBucket(bucket=self._bucket())
        pb.add(AgentData(agent_id="agent5_symptoms_mood", domain_object=1))
        pb.add(AgentData(agent_id="agent1_biomarker", domain_object=2))
        assert pb.reporting_agents == ["agent1_biomarker", "agent5_symptoms_mood"]

    def test_patient_and_bucket_id_passthrough(self):
        pb = PatientBucket(bucket=self._bucket())
        assert pb.patient_id == "p1"
        assert pb.bucket_id == "p1_2026-05-27_T2"

    def test_flare_button_default_none(self):
        pb = PatientBucket(bucket=self._bucket())
        assert pb.flare_button is None
