"""Tests for risk_engine (Phase 4 severity composite) and decision_maker (policy)."""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.conductor.decision.decision_maker import (
    ALERT_PROBABILITY,
    ALERT_SEVERITY,
    Decision,
    DecisionMaker,
)
from immunosense.conductor.fusion.risk_engine import RiskEngine
from immunosense.events.types import ConfidenceLevel


@dataclass
class _Q:
    agent_id: str
    quality: float


@dataclass
class _Conf:
    level: ConfidenceLevel
    per_agent: list = None


@dataclass
class _FakePat:
    name: str = "p"


def _out(agent_id, dim, alerts=None):
    return AgentOutput(
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        data={},
        vector=np.ones(dim),
        vector_dim=dim,
        alerts=alerts or [],
        confidence=0.9,
    )


class TestRiskEngine:
    def setup_method(self):
        self.risk = RiskEngine()

    def test_gates_when_probability_none(self):
        r = self.risk.compute(None, _Conf(ConfidenceLevel.INSUFFICIENT), {})
        assert r.severity_composite is None
        assert r.band is None
        assert r.flare_probability is None

    def test_composite_in_range(self):
        outputs = {"agent5_symptoms_mood": _out("agent5_symptoms_mood", 36)}
        r = self.risk.compute(0.5, _Conf(ConfidenceLevel.HIGH), outputs)
        assert r.severity_composite is not None
        assert 0.0 <= r.severity_composite <= 1.0

    def test_confidence_damping_low_lt_high(self):
        outputs = {
            "agent4_wearable": _out("agent4_wearable", 29, [{"severity": "critical"}]),
        }
        r_high = self.risk.compute(0.5, _Conf(ConfidenceLevel.HIGH), outputs)
        r_low = self.risk.compute(0.5, _Conf(ConfidenceLevel.LOW), outputs)
        assert r_low.severity_composite < r_high.severity_composite

    def test_band_thresholds(self):
        outputs = {"agent5_symptoms_mood": _out("agent5_symptoms_mood", 36)}
        # Low composite
        r = self.risk.compute(0.0, _Conf(ConfidenceLevel.HIGH), {})
        assert r.band == "low"

    def test_band_high_with_severe_inputs(self):
        outputs = {
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36,
                                          [{"severity": "critical"}]),
            "agent4_wearable": _out("agent4_wearable", 29,
                                     [{"severity": "critical"}]),
        }
        r = self.risk.compute(0.85, _Conf(ConfidenceLevel.HIGH), outputs)
        assert r.band in ("moderate", "high")
        assert r.severity_composite >= 0.3

    def test_no_outputs_no_acute_severity(self):
        r = self.risk.compute(0.3, _Conf(ConfidenceLevel.HIGH), {})
        assert r.acute_severity == 0.0


class TestDecisionMaker:
    def setup_method(self):
        self.dm = DecisionMaker()

    # --- Normal-confidence paths ---

    def test_high_severity_alerts_and_calls_tfm(self):
        d = self.dm.decide(
            flare_probability=0.4,
            severity_composite=0.7,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.HIGH),
            severity_band="high",
        )
        assert d.raise_alert is True
        assert d.call_tfm is True

    def test_high_probability_alerts(self):
        d = self.dm.decide(
            flare_probability=0.6,
            severity_composite=0.3,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.HIGH),
            severity_band="low",
        )
        assert d.raise_alert is True

    def test_low_everything_no_action(self):
        d = self.dm.decide(
            flare_probability=0.1,
            severity_composite=0.1,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.HIGH),
            severity_band="low",
        )
        assert d.raise_alert is False
        assert d.call_tfm is False

    def test_moderate_band_explains_without_alerting(self):
        d = self.dm.decide(
            flare_probability=0.3,
            severity_composite=0.4,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.MODERATE),
            severity_band="moderate",
        )
        assert d.raise_alert is False
        assert d.call_tfm is True

    def test_pattern_match_triggers_tfm(self):
        d = self.dm.decide(
            flare_probability=0.2,
            severity_composite=0.2,
            matched_patterns=[_FakePat("autonomic_stress")],
            confidence_result=_Conf(ConfidenceLevel.MODERATE),
            severity_band="low",
        )
        assert d.call_tfm is True

    # --- Gated (INSUFFICIENT) paths ---

    def test_insufficient_no_alert_explains_gap(self):
        d = self.dm.decide(
            flare_probability=None,
            severity_composite=None,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.INSUFFICIENT),
        )
        assert d.raise_alert is False
        assert d.call_tfm is True
        assert any("insufficient" in r.lower() for r in d.reasons)

    def test_flare_button_honored_under_insufficient(self):
        # This is the critical safety behavior: a patient pressing the button
        # raises an alert even when confidence is insufficient.
        d = self.dm.decide(
            flare_probability=None,
            severity_composite=None,
            matched_patterns=[],
            confidence_result=_Conf(ConfidenceLevel.INSUFFICIENT),
            flare_button=0.9,
        )
        assert d.raise_alert is True

    def test_explain_insufficient_can_be_disabled(self):
        dm = DecisionMaker(explain_insufficient=False)
        d = dm.decide(None, None, [], _Conf(ConfidenceLevel.INSUFFICIENT))
        assert d.call_tfm is False

    # --- Tunable thresholds ---

    def test_custom_thresholds(self):
        dm = DecisionMaker(alert_severity=0.9, alert_probability=0.9)
        # 0.7 severity wouldn't alert at 0.9 threshold
        d = dm.decide(0.5, 0.7, [], _Conf(ConfidenceLevel.HIGH), severity_band="moderate")
        assert d.raise_alert is False

    def test_default_thresholds_documented(self):
        assert ALERT_SEVERITY == 0.6
        assert ALERT_PROBABILITY == 0.5
