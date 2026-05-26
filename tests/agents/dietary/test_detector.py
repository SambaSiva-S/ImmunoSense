"""Tests for DietaryTriggerDetector."""

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.dietary.constants import BOOLEAN_TRIGGERS, CONTINUOUS_FEATURES
from immunosense.agents.dietary.detector import DietaryTriggerDetector


def _make_records(n_days=60, start_date="2026-03-01", seed=42):
    """Build n_days of baseline (no-trigger) records."""
    np_rng = np.random.RandomState(seed)
    records = []
    start = pd.Timestamp(start_date)
    for i in range(n_days):
        date = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rec = {
            "date": date,
            "dii_score": float(np_rng.normal(0.5, 1.0)),
            "omega6_omega3_ratio": float(max(1.0, np_rng.normal(10, 4))),
            "glycemic_load": float(max(20, np_rng.normal(100, 30))),
            "sodium_mg": float(max(500, np_rng.normal(2500, 800))),
            "alcohol_g": float(max(0, np_rng.exponential(5))),
            "overnight_fast_hours": float(np_rng.normal(12, 2)),
            "gluten_present": bool(np_rng.rand() < 0.55),
            "dairy_present": bool(np_rng.rand() < 0.40),
            "nightshade_present": bool(np_rng.rand() < 0.30),
            "upf_present": bool(np_rng.rand() < 0.25),
        }
        records.append(rec)
    return records


def test_detector_too_few_days_returns_empty():
    """If fewer than min_days records, return empty list."""
    detector = DietaryTriggerDetector(min_days=14)
    records = _make_records(n_days=5)
    flares = {}
    patterns = detector.detect(records, flares)
    assert patterns == []


def test_detector_no_trigger_no_flares_returns_empty():
    """No triggers + no flares should detect nothing."""
    detector = DietaryTriggerDetector(min_days=14, n_permutations=200, random_seed=42)
    records = _make_records(n_days=60, seed=42)
    flares = {}  # zero flares
    patterns = detector.detect(records, flares)
    # No variance in flares (all zero) -> nothing detected
    assert patterns == []


def test_detector_detects_planted_boolean_trigger():
    """Plant a clear boolean trigger and verify it's detected."""
    detector = DietaryTriggerDetector(min_days=14, n_permutations=500, random_seed=42)
    records = _make_records(n_days=60, seed=42)

    # Plant: every day with dairy_present=True causes a flare 2 days later
    flares = {}
    for i, rec in enumerate(records):
        if rec["dairy_present"] and i + 2 < len(records):
            flare_date = records[i + 2]["date"]
            flares[flare_date] = 2.0

    patterns = detector.detect(records, flares)
    # Should detect dairy_present at lag=2
    dairy_patterns = [
        p for p in patterns
        if p.feature == "dairy_present" and p.lag_days == 2
    ]
    assert len(dairy_patterns) >= 1
    # Should be high or medium confidence
    assert dairy_patterns[0].confidence in ("high", "medium")
    # Should have positive effect (high -> more flares)
    assert dairy_patterns[0].effect_size > 0


def test_detector_detects_planted_continuous_threshold_trigger():
    """Plant a continuous threshold trigger (sodium > 3000) and detect it."""
    detector = DietaryTriggerDetector(min_days=14, n_permutations=500, random_seed=42)
    records = _make_records(n_days=60, seed=44)

    # Plant: every day with sodium_mg > 3000 causes a flare 1 day later
    flares = {}
    for i, rec in enumerate(records):
        if rec["sodium_mg"] > 3000 and i + 1 < len(records):
            flare_date = records[i + 1]["date"]
            flares[flare_date] = 2.0

    patterns = detector.detect(records, flares)
    # Should detect sodium_mg (>p75) or similar at lag=1
    sodium_patterns = [
        p for p in patterns
        if "sodium_mg" in p.feature and p.lag_days == 1
    ]
    assert len(sodium_patterns) >= 1
    assert sodium_patterns[0].confidence in ("high", "medium")


def test_detector_respects_min_days_per_lag():
    """For a 14-day minimum, lag=10 over 20 days only has 10 pairs."""
    detector = DietaryTriggerDetector(min_days=14, lags=(0, 10), n_permutations=100)
    records = _make_records(n_days=20)
    flares = {records[15]["date"]: 1.0}
    patterns = detector.detect(records, flares)
    # lag=10 would have only 10 pairs (< min_days=14), so should be skipped
    # lag=0 has 20 pairs, eligible
    # Nothing should be detected for lag=10
    for p in patterns:
        assert p.lag_days != 10


def test_detector_confidence_tiers():
    """confidence mapping: <0.01 high, <0.05 medium, else low."""
    detector = DietaryTriggerDetector()
    assert detector._confidence(0.005) == "high"
    assert detector._confidence(0.030) == "medium"
    assert detector._confidence(0.080) == "low"
    assert detector._confidence(0.500) == "low"


def test_detector_sorts_by_effect_size():
    """Returned patterns are sorted by |effect_size| descending."""
    detector = DietaryTriggerDetector(min_days=14, n_permutations=500, random_seed=42)
    records = _make_records(n_days=60, seed=42)
    flares = {}
    for i, rec in enumerate(records):
        if rec["dairy_present"] and i + 2 < len(records):
            flares[records[i + 2]["date"]] = 2.0

    patterns = detector.detect(records, flares)
    if len(patterns) > 1:
        for i in range(len(patterns) - 1):
            assert abs(patterns[i].effect_size) >= abs(patterns[i + 1].effect_size)


def test_detector_only_positive_direction():
    """The detector should never report negative effect_size (one-sided positive only)."""
    detector = DietaryTriggerDetector(min_days=14, n_permutations=200, random_seed=42)
    records = _make_records(n_days=60, seed=42)
    flares = {records[i]["date"]: float(np.random.RandomState(i).rand()) for i in range(60)}
    patterns = detector.detect(records, flares)
    for p in patterns:
        assert p.effect_size > 0, f"Negative effect detected: {p.feature}={p.effect_size}"


def test_detector_deterministic():
    """Same seed -> same patterns."""
    records = _make_records(n_days=60, seed=42)
    flares = {records[i]["date"]: 1.0 if i % 3 == 0 else 0.0 for i in range(60)}

    det1 = DietaryTriggerDetector(min_days=14, n_permutations=200, random_seed=42)
    det2 = DietaryTriggerDetector(min_days=14, n_permutations=200, random_seed=42)

    p1 = det1.detect(records, flares)
    p2 = det2.detect(records, flares)

    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert a.feature == b.feature
        assert a.lag_days == b.lag_days
        assert abs(a.effect_size - b.effect_size) < 1e-9
