"""End-to-end DietaryAgent tests with synthetic patients."""

import random as random_mod
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.agents.dietary import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
    DailyRollup,
    DietaryAgent,
)


def _rollup_from_dict(rec: dict) -> DailyRollup:
    """Build a minimal DailyRollup from a flat feature dict."""
    return DailyRollup(
        date=rec["date"],
        meal_count=4,
        daily_nutrients={},
        dii_score=rec["dii_score"],
        omega6_omega3_ratio=rec["omega6_omega3_ratio"],
        glycemic_load=rec["glycemic_load"],
        sodium_mg=rec["sodium_mg"],
        alcohol_g=rec["alcohol_g"],
        first_meal_timestamp=rec.get("first_meal_timestamp"),
        last_meal_timestamp=rec.get("last_meal_timestamp"),
        longest_intraday_gap_hours=float("nan"),
        gluten_present=rec["gluten_present"],
        dairy_present=rec["dairy_present"],
        nightshade_present=rec["nightshade_present"],
        upf_present=rec["upf_present"],
        feature_confidence={"dii_score": "high"},
        daily_dii_percentile=None,
        meal_results=[],
    )


def _generate_synthetic_records(
    n_days: int = 60,
    seed: int = 42,
    trigger: str | None = None,
    trigger_flare_prob: float = 0.70,
    baseline_flare_prob: float = 0.10,
    flare_lag_days: int = 2,
) -> tuple:
    """Replicate the SyntheticPatient generator from the original notebook."""
    rng = random_mod.Random(seed)
    np_rng = np.random.RandomState(seed)
    start = pd.Timestamp("2026-03-01")
    records = []
    scheduled_flares = {}

    for i in range(n_days):
        date = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rec = {
            "date": date,
            "dii_score": float(np_rng.normal(0.5, 1.0)),
            "omega6_omega3_ratio": float(max(1.0, np_rng.normal(10, 4))),
            "glycemic_load": float(max(20, np_rng.normal(100, 30))),
            "sodium_mg": float(max(500, np_rng.normal(2500, 800))),
            "alcohol_g": float(max(0, np_rng.exponential(5))),
            "gluten_present": rng.random() < 0.55,
            "dairy_present": rng.random() < 0.40,
            "nightshade_present": rng.random() < 0.30,
            "upf_present": rng.random() < 0.25,
            "first_meal_timestamp": f"{date}T08:00:00",
            "last_meal_timestamp": f"{date}T19:00:00",
        }

        is_trigger_active = False
        if trigger in BOOLEAN_TRIGGERS:
            is_trigger_active = rec[trigger]
        elif trigger == "sodium_mg":
            is_trigger_active = rec["sodium_mg"] > 3000
        elif trigger == "glycemic_load":
            is_trigger_active = rec["glycemic_load"] > 120
        elif trigger == "alcohol_g":
            is_trigger_active = rec["alcohol_g"] > 15

        flare_prob = trigger_flare_prob if is_trigger_active else baseline_flare_prob
        if rng.random() < flare_prob:
            flare_date = (start + pd.Timedelta(days=i + flare_lag_days)).strftime("%Y-%m-%d")
            severity = float(np_rng.uniform(1.5, 3.0))
            scheduled_flares[flare_date] = max(scheduled_flares.get(flare_date, 0.0), severity)

        records.append(rec)

    return records, scheduled_flares


# ============================================================
# Basic agent API
# ============================================================

def test_agent_init():
    agent = DietaryAgent(patient_id="p001")
    assert agent.agent_id == "agent2_dietary"
    assert agent.output_dim == 10
    assert agent.patient_id == "p001"


def test_agent_observe_returns_anomalies():
    agent = DietaryAgent()
    rec = _generate_synthetic_records(n_days=1, seed=42)[0][0]
    rollup = _rollup_from_dict(rec)
    out = agent.observe(rollup)
    assert "overnight_fast_hours" in out
    assert "anomaly_scores" in out


def test_agent_observe_flare():
    agent = DietaryAgent()
    agent.observe_flare("2026-04-01", 2.5)
    assert agent.flare_events["2026-04-01"] == 2.5


def test_agent_analyze_too_few_days():
    """Agent.analyze() with <14 days returns empty patterns."""
    agent = DietaryAgent(trigger_min_days=14)
    records, _ = _generate_synthetic_records(n_days=5, seed=42)
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    report = agent.analyze()
    assert report.detected_patterns == []


# ============================================================
# Process (BaseAgent interface)
# ============================================================

def test_agent_process_returns_agent_output():
    agent = DietaryAgent(patient_id="p001")
    rec = _generate_synthetic_records(n_days=1, seed=42)[0][0]
    rollup = _rollup_from_dict(rec)
    out = agent.process({"rollup": rollup})
    assert isinstance(out, AgentOutput)
    assert out.agent_id == "agent2_dietary"
    assert out.vector_dim == 10
    assert out.vector.shape == (10,)


def test_agent_process_missing_rollup_raises():
    agent = DietaryAgent()
    with pytest.raises(ValueError, match="requires input_data\\['rollup'\\]"):
        agent.process({})


def test_agent_process_records_latency():
    agent = DietaryAgent()
    rec = _generate_synthetic_records(n_days=1, seed=42)[0][0]
    agent.process({"rollup": _rollup_from_dict(rec)})
    assert len(agent._latencies) == 1
    assert agent._latencies[0] >= 0


def test_agent_vector_first_day_is_zeros():
    """First day: anomaly scores are NaN -> become 0.0. Booleans cast to 0/1."""
    agent = DietaryAgent()
    rec = _generate_synthetic_records(n_days=1, seed=42)[0][0]
    rec["dairy_present"] = True
    rec["gluten_present"] = False
    agent.observe(_rollup_from_dict(rec))
    vec = agent.get_output_vector()
    # All continuous anomalies should be 0.0 (NaN -> 0)
    for i in range(len(CONTINUOUS_FEATURES)):
        assert vec[i] == 0.0
    # gluten_present is BOOLEAN_TRIGGERS[0], dairy_present is BOOLEAN_TRIGGERS[1]
    n_cont = len(CONTINUOUS_FEATURES)
    assert vec[n_cont + 0] == 0.0  # gluten False
    assert vec[n_cont + 1] == 1.0  # dairy True


# ============================================================
# Trigger detection validation (the gold-standard test from the notebook)
# ============================================================

@pytest.mark.slow
def test_agent_detects_dairy_trigger():
    """Bullseye: planted DAIRY trigger at lag=2 with p=0.70 should be detected."""
    agent = DietaryAgent(window=14, n_permutations=500, random_seed=42)
    records, flares = _generate_synthetic_records(
        n_days=60, seed=42,
        trigger="dairy_present", trigger_flare_prob=0.70,
        baseline_flare_prob=0.10, flare_lag_days=2,
    )
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    for date, severity in flares.items():
        agent.observe_flare(date, severity)

    report = agent.analyze()

    # Should find dairy_present at lag=2 with high confidence
    bullseye = [
        p for p in report.detected_patterns
        if p.feature == "dairy_present" and p.lag_days == 2
    ]
    assert len(bullseye) >= 1, (
        f"Expected dairy_present at lag=2; got: "
        f"{[(p.feature, p.lag_days, p.confidence) for p in report.detected_patterns]}"
    )
    assert bullseye[0].confidence in ("high", "medium")
    assert bullseye[0].effect_size > 0


@pytest.mark.slow
def test_agent_detects_sodium_trigger():
    """Planted SODIUM trigger (>3000mg, lag=1) should be detected."""
    agent = DietaryAgent(window=14, n_permutations=500, random_seed=42)
    records, flares = _generate_synthetic_records(
        n_days=60, seed=44,
        trigger="sodium_mg", trigger_flare_prob=0.70,
        baseline_flare_prob=0.10, flare_lag_days=1,
    )
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    for date, severity in flares.items():
        agent.observe_flare(date, severity)

    report = agent.analyze()

    # Should find sodium_mg at lag=1 with high confidence
    sodium = [
        p for p in report.detected_patterns
        if "sodium_mg" in p.feature and p.lag_days == 1
    ]
    assert len(sodium) >= 1
    assert sodium[0].confidence in ("high", "medium")
    assert sodium[0].effect_size > 0


# ============================================================
# Report structure
# ============================================================

def test_report_includes_baselines():
    agent = DietaryAgent(trigger_min_days=14)
    records, _ = _generate_synthetic_records(n_days=20, seed=42)
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    report = agent.analyze()

    assert "continuous" in report.baselines
    assert "boolean" in report.baselines
    for f in CONTINUOUS_FEATURES:
        assert f in report.baselines["continuous"]
    for f in BOOLEAN_TRIGGERS:
        assert f in report.baselines["boolean"]


def test_report_activation_flags():
    agent = DietaryAgent(trigger_min_days=14)
    records, _ = _generate_synthetic_records(n_days=20, seed=42)
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    report = agent.analyze()

    assert "baseline_continuous" in report.tracker_activation
    assert "trigger_detector" in report.tracker_activation
    # 20 days >= min_days=14 -> trigger_detector should be activated
    assert report.tracker_activation["trigger_detector"] is True


def test_report_counts_flare_events():
    agent = DietaryAgent()
    records, _ = _generate_synthetic_records(n_days=10, seed=42)
    for rec in records:
        agent.observe(_rollup_from_dict(rec))
    agent.observe_flare("2026-03-05", 2.0)
    agent.observe_flare("2026-03-06", 1.5)
    agent.observe_flare("2026-03-07", 0.0)  # zero severity, not counted
    report = agent.analyze()
    assert report.n_flare_events == 2


def test_agent_confidence_aggregation():
    """Agent.process confidence reflects rollup feature_confidence."""
    agent = DietaryAgent()
    rec = _generate_synthetic_records(n_days=1, seed=42)[0][0]
    rollup = _rollup_from_dict(rec)
    # Override the feature_confidence with all 'high'
    rollup.feature_confidence = {f: "high" for f in ["dii_score", "sodium_mg", "alcohol_g"]}
    out = agent.process({"rollup": rollup})
    assert out.confidence == 1.0
