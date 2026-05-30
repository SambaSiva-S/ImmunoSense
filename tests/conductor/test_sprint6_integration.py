"""End-to-end Sprint 6 integration: full Conductor pipeline with real components.

Verifies that fusion + corroboration + risk + decision + TFM + embedding all
talk to each other correctly through the Conductor, and that the produced
ConductorReport is internally consistent.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.adapters import (
    AdapterRegistry,
    SymptomsMoodAdapter,
    WearableAdapter,
)
from immunosense.agents.base import AgentOutput, BaseAgent
from immunosense.conductor import Conductor
from immunosense.events import (
    AgentData,
    BucketBuilder,
    EventLog,
    EventType,
    UserBucket,
)


class _ElevatedAgent(BaseAgent):
    """A fake agent that always returns elevated signals (high confidence + critical alert)."""

    def __init__(self, agent_id, dim, poll):
        super().__init__()
        self.agent_id = agent_id
        self.output_dim = dim
        self.poll_frequency = poll

    def process(self, input_data):
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"ok": True},
            vector=np.ones(self.output_dim) * 0.8,
            vector_dim=self.output_dim,
            alerts=[{"severity": "critical", "name": "test"}],
            confidence=0.9,
        )


class _QuietAgent(BaseAgent):
    """A fake agent with no alerts and low signal."""

    def __init__(self, agent_id, dim, poll):
        super().__init__()
        self.agent_id = agent_id
        self.output_dim = dim
        self.poll_frequency = poll

    def process(self, input_data):
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"ok": True},
            vector=np.zeros(self.output_dim),
            vector_dim=self.output_dim,
            alerts=[],
            confidence=0.9,
        )


@pytest.fixture
def two_elevated(tmp_path):
    """Conductor with two elevated agents (symptoms + wearable)."""
    sym = _ElevatedAgent("agent5_symptoms_mood", 36, "daily")
    wear = _ElevatedAgent("agent4_wearable", 29, "1hr")
    registry = AdapterRegistry()
    registry.register(SymptomsMoodAdapter(sym))
    registry.register(WearableAdapter(wear))
    log = EventLog(tmp_path)
    conductor = Conductor(registry=registry, event_log=log, disease="SLE")
    ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
    bucket = BucketBuilder.bucket_for("patient001", ts)
    pb = UserBucket(bucket=bucket)
    pb.add(AgentData("agent5_symptoms_mood", "fake_summary", produced_at=ts))
    pb.add(AgentData(
        "agent4_wearable",
        {"night_df": "df", "rr_intervals": [800, 810], "night_idx": 1},
        produced_at=ts,
    ))
    return conductor, log, pb, bucket


class TestFullPipelineElevated:
    def test_report_is_internally_consistent(self, two_elevated):
        conductor, log, pb, bucket = two_elevated
        report = conductor.evaluate_bucket(pb)

        # Probability + composite both present (not gated).
        assert report.flare_probability is not None
        assert report.flare_probability > 0.08  # above baseline prior
        assert report.severity_composite is not None
        assert 0.0 <= report.severity_composite <= 1.0

        # Confidence at least LOW.
        assert report.confidence_level.value in ("low", "moderate", "high")

        # Calibration version recorded.
        assert report.calibration_version == "lr-v1"

        # Embedding envelope has stable layout.
        assert report.embedding_concat_dim == 87

        # Decision is consistent with severity.
        assert report.decision is not None
        if report.severity_composite >= 0.6:
            assert report.decision.raise_alert is True

    def test_corroboration_fires(self, two_elevated):
        conductor, log, pb, bucket = two_elevated
        report = conductor.evaluate_bucket(pb)
        names = [p.name for p in report.matched_patterns]
        # symptoms + wearable both elevated -> autonomic_stress must match.
        assert "autonomic_stress" in names

    def test_tfm_called_and_safe(self, two_elevated):
        conductor, log, pb, bucket = two_elevated
        report = conductor.evaluate_bucket(pb)
        # A pattern matched -> decision should call TFM.
        assert report.decision.call_tfm is True
        assert report.explanation is not None
        assert report.tfm_ok is True
        text = report.explanation.lower()
        # Safety language present (non-prescriptive).
        assert "clinician" in text or "diagnosis" in text or "medical" in text

    def test_no_double_counting(self, two_elevated):
        """Phase 2 patterns must not have a probability field that could feed Phase 1."""
        conductor, log, pb, bucket = two_elevated
        report = conductor.evaluate_bucket(pb)
        for p in report.matched_patterns:
            assert not hasattr(p, "probability")
            assert not hasattr(p, "lr")
            assert not hasattr(p, "likelihood_ratio")

    def test_bucket_eval_event_serializable(self, two_elevated):
        import json
        conductor, log, pb, bucket = two_elevated
        report = conductor.evaluate_bucket(pb)
        evt = [e for e in log.read_bucket(pb.user_id, pb.bucket_id)
               if e.event_type == EventType.BUCKET_EVAL][0]
        # Round-trips through JSON cleanly.
        s = json.dumps(evt.payload)
        recovered = json.loads(s)
        assert recovered["flare_probability"] == report.flare_probability
        assert recovered["calibration_version"] == "lr-v1"
        assert recovered["raised_alert"] == report.decision.raise_alert


class TestFullPipelineGated:
    def test_insufficient_gates_probability(self, tmp_path):
        """With one quiet agent only, confidence is INSUFFICIENT -> probability None."""
        sym = _QuietAgent("agent5_symptoms_mood", 36, "daily")
        # Make it report confidence 0.0 so quality is 0 too.
        original = sym.process
        def q_process(input_data):
            out = original(input_data)
            out.confidence = 0.0
            return out
        sym.process = q_process

        registry = AdapterRegistry()
        registry.register(SymptomsMoodAdapter(sym))
        log = EventLog(tmp_path)
        conductor = Conductor(registry=registry, event_log=log, disease="SLE")
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        bucket = BucketBuilder.bucket_for("p1", ts)
        pb = UserBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "s", produced_at=ts))

        report = conductor.evaluate_bucket(pb)
        # Probability and composite gated.
        assert report.flare_probability is None
        assert report.severity_composite is None
        assert report.severity_band is None
        # Decision still runs, TFM explains the gap.
        assert report.decision.raise_alert is False
        assert report.decision.call_tfm is True
        assert report.explanation is not None
        # Embedding envelope still built.
        assert report.embedding_concat_dim == 87
