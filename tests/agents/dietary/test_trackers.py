"""Tests for dietary trackers: OvernightFastTracker, DietaryRobustTracker."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.dietary.constants import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
)
from immunosense.agents.dietary.trackers import (
    DietaryRobustTracker,
    OvernightFastTracker,
    _UnivariateRobustTracker,
)


# ============================================================
# OvernightFastTracker
# ============================================================

def _rollup(first: str | None, last: str | None) -> SimpleNamespace:
    """Build a minimal rollup-like object."""
    return SimpleNamespace(first_meal_timestamp=first, last_meal_timestamp=last)


def test_overnight_fast_first_day_is_nan():
    """First observation has no 'yesterday', returns NaN."""
    t = OvernightFastTracker()
    out = t.compute(_rollup("2026-04-01T08:00:00", "2026-04-01T19:00:00"))
    assert pd.isna(out)


def test_overnight_fast_second_day_computes():
    """Day 1 sets state, Day 2 computes gap between yesterday's last and today's first."""
    t = OvernightFastTracker()
    t.compute(_rollup("2026-04-01T08:00:00", "2026-04-01T19:00:00"))
    # Yesterday last: 19:00. Today first: 07:30 -> 12.5 hours
    out = t.compute(_rollup("2026-04-02T07:30:00", "2026-04-02T19:00:00"))
    assert abs(out - 12.5) < 1e-6


def test_overnight_fast_missing_today_returns_nan():
    """If today's first_meal_timestamp is None, returns NaN."""
    t = OvernightFastTracker()
    t.compute(_rollup("2026-04-01T08:00:00", "2026-04-01T19:00:00"))
    out = t.compute(_rollup(None, "2026-04-02T19:00:00"))
    assert pd.isna(out)


def test_overnight_fast_missing_yesterday_returns_nan():
    """If yesterday had no last_meal_timestamp, today's gap is undefined."""
    t = OvernightFastTracker()
    t.compute(_rollup("2026-04-01T08:00:00", None))
    out = t.compute(_rollup("2026-04-02T07:30:00", "2026-04-02T19:00:00"))
    assert pd.isna(out)


def test_overnight_fast_invalid_gap_returns_nan():
    """A 50-hour gap (>48 cap) returns NaN."""
    t = OvernightFastTracker()
    t.compute(_rollup("2026-04-01T08:00:00", "2026-04-01T19:00:00"))
    # 3 days later (gap = 60 hours)
    out = t.compute(_rollup("2026-04-04T07:00:00", "2026-04-04T19:00:00"))
    assert pd.isna(out)


def test_overnight_fast_negative_gap_returns_nan():
    """A negative gap (timestamp ordering broken) returns NaN."""
    t = OvernightFastTracker()
    t.compute(_rollup("2026-04-01T08:00:00", "2026-04-02T22:00:00"))
    # Today first = yesterday's last - 4 hours (impossible scenario, but defensive)
    out = t.compute(_rollup("2026-04-02T18:00:00", "2026-04-02T19:00:00"))
    assert pd.isna(out)


# ============================================================
# _UnivariateRobustTracker
# ============================================================

def test_univariate_tracker_basic():
    """Basic median + IQR tracking."""
    t = _UnivariateRobustTracker(window=14)
    for v in [10, 12, 11, 13, 9, 14, 10]:
        t.update(v)
    b = t.baseline()
    # Outlier exclusion may drop a few values; just verify enough remain for stats
    assert b["n_clean"] >= 3
    assert b["n_all"] == 7  # all 7 observations recorded in all_history
    assert 9 <= b["median"] <= 14


def test_univariate_tracker_skips_nan():
    """NaN values should not be recorded."""
    t = _UnivariateRobustTracker()
    t.update(10)
    t.update(np.nan)
    t.update(12)
    assert len(t.all_history) == 2


def test_univariate_tracker_excludes_outliers():
    """Values >2 IQRs from median should be excluded from clean_history."""
    t = _UnivariateRobustTracker(window=20)
    # Build a tight baseline at 10±1
    for v in [10, 10, 10, 11, 9, 10, 11, 9, 10]:
        t.update(v)
    # Now feed in a huge outlier
    t.update(1000)
    # The outlier should be in all_history but NOT in clean_history
    assert 1000.0 in list(t.all_history)
    assert 1000.0 not in list(t.clean_history)


def test_univariate_tracker_anomaly_score_nan_when_no_baseline():
    """anomaly_score requires >=3 clean observations."""
    t = _UnivariateRobustTracker()
    t.update(10)
    t.update(11)
    # Only 2 observations
    assert pd.isna(t.anomaly_score(15))


def test_univariate_tracker_anomaly_score_normal():
    """anomaly_score = (value - median) / IQR."""
    t = _UnivariateRobustTracker(window=20)
    # Build a tight baseline where no value is rejected as outlier
    for v in [10, 11, 9, 12, 10, 11, 8, 12, 10, 11]:
        t.update(v)
    # Now score a value 2 IQRs above the median
    baseline = t.baseline()
    iqr = baseline["iqr"]
    median = baseline["median"]
    test_value = median + (iqr * 2.0)
    score = t.anomaly_score(test_value)
    # Should be approximately 2.0
    assert 1.5 < score < 2.5


def test_univariate_tracker_personalization_weight():
    """personal_weight grows with clean_history size, capped at 1.0."""
    t = _UnivariateRobustTracker(window=30, personalization_days=10)
    # Use tight baseline values that don't get rejected as outliers
    for _ in range(5):
        t.update(100.0)
    weight_5 = t.baseline()["personal_weight"]

    # Add 10 more (also tight, no outliers); cumulative will be 15
    for _ in range(10):
        t.update(100.0)
    weight_15 = t.baseline()["personal_weight"]

    assert weight_5 < 1.0
    assert weight_15 == 1.0  # capped at 1.0 once n_clean >= personalization_days


# ============================================================
# DietaryRobustTracker
# ============================================================

def test_dietary_tracker_initializes_all_features():
    """Should initialize trackers for all CONTINUOUS_FEATURES and BOOLEAN_TRIGGERS."""
    t = DietaryRobustTracker()
    for f in CONTINUOUS_FEATURES:
        assert f in t.continuous_trackers
    for f in BOOLEAN_TRIGGERS:
        assert f in t.boolean_history


def test_dietary_tracker_update_dispatches():
    """update() should populate per-feature trackers from a daily dict."""
    t = DietaryRobustTracker()
    t.update({
        "dii_score": 0.5,
        "sodium_mg": 2500,
        "dairy_present": True,
        "gluten_present": False,
    })
    assert 0.5 in t.continuous_trackers["dii_score"].all_history
    assert 2500.0 in t.continuous_trackers["sodium_mg"].all_history
    assert True in t.boolean_history["dairy_present"]
    assert False in t.boolean_history["gluten_present"]


def test_dietary_tracker_report_shape():
    """report() should return continuous + boolean sub-dicts."""
    t = DietaryRobustTracker()
    for day in range(5):
        t.update({
            "dii_score": 0.5 + 0.1 * day,
            "sodium_mg": 2500,
            "dairy_present": day % 2 == 0,
        })
    rep = t.report()
    assert "continuous" in rep
    assert "boolean" in rep
    for f in CONTINUOUS_FEATURES:
        assert f in rep["continuous"]
    for f in BOOLEAN_TRIGGERS:
        assert f in rep["boolean"]


def test_dietary_tracker_boolean_prevalence():
    """Boolean prevalence = mean of (True/False) values."""
    t = DietaryRobustTracker()
    for v in [True, True, False, True, False]:  # 3/5 = 0.6 prevalence
        t.update({"dairy_present": v})
    rep = t.report()
    assert abs(rep["boolean"]["dairy_present"]["prevalence"] - 0.6) < 1e-9


def test_dietary_tracker_anomaly_scores_shape():
    """anomaly_scores returns one entry per CONTINUOUS_FEATURES."""
    t = DietaryRobustTracker()
    for day in range(5):
        t.update({f: 0.5 + 0.1 * day for f in CONTINUOUS_FEATURES})
    scores = t.anomaly_scores({f: 0.5 for f in CONTINUOUS_FEATURES})
    for f in CONTINUOUS_FEATURES:
        assert f in scores


def test_dietary_tracker_missing_field_handled():
    """update() with partial feature dict should not crash."""
    t = DietaryRobustTracker()
    t.update({"dii_score": 0.5})  # missing every other feature
    # Should still report sane baseline structure
    rep = t.report()
    assert rep["continuous"]["dii_score"]["n_all"] == 1
    assert rep["continuous"]["sodium_mg"]["n_all"] == 0
