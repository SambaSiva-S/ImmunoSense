"""Tests for Bayesian fusion (Challenge 3 Phase 1).

Verifies the math properties that make the fusion correct:
gating, prior preservation, elevation, quality tempering, compounding evidence,
exact log-odds round-trip, and reassurance.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.conductor.calibration import (
    CALIBRATION_VERSION,
    DEFAULT_BASELINE_FLARE_PRIOR,
    load_calibration,
)
from immunosense.conductor.fusion.statistical_fusion import (
    StatisticalFusion,
    _log_odds_to_prob,
    _prob_to_log_odds,
    extract_signal_strength,
)
from immunosense.events.types import ConfidenceLevel


@dataclass
class _Q:
    agent_id: str
    quality: float


@dataclass
class _Conf:
    level: ConfidenceLevel
    per_agent: list


def _out(agent_id, dim, alerts=None, vec_scale=1.0):
    return AgentOutput(
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        data={},
        vector=np.ones(dim) * vec_scale,
        vector_dim=dim,
        alerts=alerts or [],
        confidence=0.9,
    )


_CRIT = [{"severity": "critical"}]


class TestCalibration:
    def test_default_version_loads(self):
        cal = load_calibration()
        assert cal.version == CALIBRATION_VERSION
        assert cal.baseline_prior == DEFAULT_BASELINE_FLARE_PRIOR
        assert "agent5_symptoms_mood" in cal.agent_ids
        assert "agent1_biomarker" in cal.agent_ids

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError):
            load_calibration("not-a-real-version")

    def test_lrs_are_provisional_with_sources(self):
        cal = load_calibration()
        for agent_id in cal.agent_ids:
            lr = cal.get(agent_id)
            assert lr.source, f"{agent_id} missing provenance"
            assert lr.lr_positive > 0
            assert lr.lr_negative > 0


class TestLogOddsRoundTrip:
    @pytest.mark.parametrize("p", [0.05, 0.08, 0.5, 0.7, 0.95])
    def test_round_trip(self, p):
        assert _log_odds_to_prob(_prob_to_log_odds(p)) == pytest.approx(p, abs=1e-9)


class TestSignalExtraction:
    def test_no_alerts_no_vector(self):
        out = AgentOutput(
            agent_id="x", timestamp=datetime.now(timezone.utc),
            data={}, vector=None, vector_dim=0, alerts=[], confidence=0,
        )
        assert extract_signal_strength(out) == 0.0

    def test_critical_alert_strong_signal(self):
        out = _out("a", 5, alerts=[{"severity": "critical"}], vec_scale=0.0)
        assert extract_signal_strength(out) > 0.6

    def test_warning_alert_weaker_than_critical(self):
        crit = extract_signal_strength(_out("a", 5, alerts=[{"severity": "critical"}]))
        warn = extract_signal_strength(_out("a", 5, alerts=[{"severity": "warning"}]))
        assert crit > warn

    def test_alert_key_normalization(self):
        # symptoms agent uses 'severity', biomarker uses 'level' — both work.
        s1 = extract_signal_strength(_out("a", 5, alerts=[{"severity": "critical"}]))
        s2 = extract_signal_strength(_out("a", 5, alerts=[{"level": "critical"}]))
        assert s1 == s2


class TestFusionBehavior:
    def setup_method(self):
        self.fusion = StatisticalFusion()
        self.prior = DEFAULT_BASELINE_FLARE_PRIOR

    def test_insufficient_confidence_gates(self):
        r = self.fusion.fuse({}, _Conf(ConfidenceLevel.INSUFFICIENT, []))
        assert r.flare_probability is None
        assert r.gated is True
        assert r.contributions == []

    def test_no_signal_stays_near_prior(self):
        out = _out("agent5_symptoms_mood", 36, vec_scale=0.3)  # mild signal -> neutral
        conf = _Conf(ConfidenceLevel.MODERATE, [_Q("agent5_symptoms_mood", 0.9)])
        r = self.fusion.fuse({"agent5_symptoms_mood": out}, conf)
        assert abs(r.flare_probability - self.prior) < 0.05

    def test_elevated_signal_raises_probability(self):
        out = _out("agent5_symptoms_mood", 36, alerts=_CRIT)
        conf = _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 1.0)])
        r = self.fusion.fuse({"agent5_symptoms_mood": out}, conf)
        assert r.flare_probability > self.prior
        assert r.contributions[0].direction == "elevated"

    def test_reassuring_signal_lowers_probability(self):
        # No alerts + zero vector -> signal 0 <= low_threshold -> reassuring.
        out = _out("agent1_biomarker", 7, alerts=[], vec_scale=0.0)
        conf = _Conf(ConfidenceLevel.HIGH, [_Q("agent1_biomarker", 1.0)])
        r = self.fusion.fuse({"agent1_biomarker": out}, conf)
        assert r.contributions[0].direction == "reassuring"
        assert r.flare_probability < self.prior

    def test_quality_tempers_lr(self):
        out = _out("agent5_symptoms_mood", 36, alerts=_CRIT)
        full = self.fusion.fuse(
            {"agent5_symptoms_mood": out},
            _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 1.0)]),
        )
        half = self.fusion.fuse(
            {"agent5_symptoms_mood": out},
            _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 0.5)]),
        )
        zero = self.fusion.fuse(
            {"agent5_symptoms_mood": out},
            _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 0.0)]),
        )
        assert full.flare_probability > half.flare_probability > self.prior
        assert abs(zero.flare_probability - self.prior) < 1e-6  # 0 quality -> no update

    def test_compounding_evidence(self):
        outs = {
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, alerts=_CRIT),
            "agent4_wearable": _out("agent4_wearable", 29, alerts=_CRIT),
        }
        conf = _Conf(
            ConfidenceLevel.HIGH,
            [_Q("agent5_symptoms_mood", 1.0), _Q("agent4_wearable", 1.0)],
        )
        r_both = self.fusion.fuse(outs, conf)
        r_one = self.fusion.fuse(
            {"agent5_symptoms_mood": outs["agent5_symptoms_mood"]},
            _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 1.0)]),
        )
        assert r_both.flare_probability > r_one.flare_probability

    def test_unknown_agent_contributes_nothing(self):
        out = _out("agent99_mystery", 5, alerts=_CRIT)
        conf = _Conf(ConfidenceLevel.HIGH, [_Q("agent99_mystery", 1.0)])
        r = self.fusion.fuse({"agent99_mystery": out}, conf)
        # No calibration entry -> no contribution -> stays at prior.
        assert abs(r.flare_probability - self.prior) < 1e-6
        assert r.contributions == []

    def test_calibration_version_recorded(self):
        r = self.fusion.fuse({}, _Conf(ConfidenceLevel.MODERATE, []))
        assert r.calibration_version == CALIBRATION_VERSION

    def test_probability_bounded(self):
        # 5 agents, all critical, full quality — must still produce prob in [0,1].
        outs = {
            f"agent{i}_x": _out(f"agent{i}_x", 5, alerts=_CRIT)
            for i in range(5)
        }
        # Use real agent ids so they actually contribute.
        outs = {
            "agent1_biomarker": _out("agent1_biomarker", 7, alerts=_CRIT),
            "agent2_dietary": _out("agent2_dietary", 10, alerts=_CRIT),
            "agent3_environment": _out("agent3_environment", 5, alerts=_CRIT),
            "agent4_wearable": _out("agent4_wearable", 29, alerts=_CRIT),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, alerts=_CRIT),
        }
        conf = _Conf(
            ConfidenceLevel.HIGH,
            [_Q(aid, 1.0) for aid in outs],
        )
        r = self.fusion.fuse(outs, conf)
        assert 0.0 <= r.flare_probability <= 1.0
