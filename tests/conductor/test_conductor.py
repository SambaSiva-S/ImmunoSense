"""Tests for the Conductor orchestration flow (Sprint 5)."""

from datetime import datetime, timezone

import pytest

from immunosense.adapters import AdapterRegistry
from immunosense.conductor import Conductor
from immunosense.conductor.utils.validation import BucketValidationError
from immunosense.events import (
    AgentData,
    BucketBuilder,
    EventLog,
    EventType,
    PatientBucket,
)
from immunosense.events.types import ConfidenceLevel


@pytest.fixture
def setup(tmp_path, fake_biomarker, fake_symptoms):
    registry = AdapterRegistry.from_agents([fake_biomarker, fake_symptoms])
    log = EventLog(tmp_path)
    conductor = Conductor(registry=registry, event_log=log)
    ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
    bucket = BucketBuilder.bucket_for("p1", ts)
    return conductor, log, bucket, ts


class TestConductorFlow:
    def test_both_agents_report(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent1_biomarker",
                         {"demographics": {}, "reading": {}}, produced_at=ts))
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))

        report = conductor.evaluate_bucket(pb)
        assert report.reporting_agents == ["agent1_biomarker", "agent5_symptoms_mood"]
        assert len(report.errors) == 0
        assert report.agent_quality["agent1_biomarker"].ok
        assert report.agent_quality["agent5_symptoms_mood"].ok

    def test_sprint6_inference_runs(self, setup):
        # With a single zero-confidence fake agent, confidence is INSUFFICIENT,
        # so fusion gates probability to None — but the decision layer still
        # calls the TFM to explain the 'not enough data' state, and an
        # explanation is produced (via the default MockTFM).
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        # Gated: insufficient confidence -> no probability/composite.
        assert report.flare_probability is None
        assert report.severity_composite is None
        # But the decision + TFM still ran to explain the gap.
        assert report.decision is not None
        assert report.decision.call_tfm is True
        assert report.explanation is not None
        assert report.tfm_ok is True
        # Embedding envelope assembled regardless.
        assert report.embedding_concat_dim == 87
        assert report.calibration_version == "lr-v1"

    def test_one_agent_fails_other_continues(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        # Biomarker gets garbage -> fails; symptoms is fine.
        pb.add(AgentData("agent1_biomarker", {"garbage": 1}, produced_at=ts))
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))

        report = conductor.evaluate_bucket(pb)
        assert len(report.errors) == 1
        assert report.agent_quality["agent1_biomarker"].ok is False
        assert report.agent_quality["agent1_biomarker"].quality == 0.0
        # Symptoms still ran.
        assert report.agent_quality["agent5_symptoms_mood"].ok is True

    def test_absent_agent_recorded(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        # Only symptoms reports; biomarker is absent.
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        # Biomarker still appears in agent_quality, marked not reported.
        assert "agent1_biomarker" in report.agent_quality
        assert report.agent_quality["agent1_biomarker"].reported is False

    def test_layer_a_events_emitted(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent1_biomarker",
                         {"demographics": {}, "reading": {}}, produced_at=ts))
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        conductor.evaluate_bucket(pb)

        events = log.read_bucket("p1", bucket.bucket_id)
        types = [e.event_type for e in events]
        assert types.count(EventType.AGENT_OUTPUT) == 2
        assert types.count(EventType.BUCKET_EVAL) == 1

    def test_agent_error_event_on_failure(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent1_biomarker", {"garbage": 1}, produced_at=ts))
        conductor.evaluate_bucket(pb)
        events = log.read_bucket("p1", bucket.bucket_id)
        error_events = [e for e in events if e.event_type == EventType.AGENT_ERROR]
        assert len(error_events) == 1
        assert "error" in error_events[0].payload

    def test_all_events_share_trace_id(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        events = log.read_bucket("p1", bucket.bucket_id)
        trace_ids = {e.trace_id for e in events}
        assert len(trace_ids) == 1
        assert report.trace_id in trace_ids

    def test_report_summary_serializable(self, setup):
        import json
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        # summary() must be JSON-serializable (it's stored in a BUCKET_EVAL event).
        json.dumps(report.summary())


class TestFlareButtonOverride:
    def test_flare_button_logs_and_reevaluates(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))

        report = conductor.on_flare_button(pb, severity=0.9)
        assert pb.flare_button == 0.9

        events = log.read_bucket("p1", bucket.bucket_id)
        flare_events = [e for e in events if e.event_type == EventType.FLARE_BUTTON]
        assert len(flare_events) == 1
        assert flare_events[0].payload["severity"] == 0.9
        assert flare_events[0].quality == 1.0


class TestValidation:
    def test_invalid_flare_severity_raises(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket, flare_button=1.5)  # out of range
        with pytest.raises(BucketValidationError):
            conductor.evaluate_bucket(pb)

    def test_unregistered_agent_warns_not_fatal(self, setup):
        conductor, log, bucket, ts = setup
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent99_unknown", "data", produced_at=ts))
        # Should not raise; should warn and skip.
        report = conductor.evaluate_bucket(pb)
        assert any("agent99_unknown" in w for w in report.warnings)
