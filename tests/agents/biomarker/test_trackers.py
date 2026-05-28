"""Tests for Layer 3 BiomarkerBaselineTracker."""

import numpy as np
import pytest

from immunosense.agents.biomarker.layer3.trackers import BiomarkerBaselineTracker


@pytest.fixture
def tracker():
    return BiomarkerBaselineTracker(biomarkers=["CRP", "ESR"])


def test_no_personal_data_at_start(tracker):
    ctx = tracker.get_personal_context({"CRP": 2.0})
    assert ctx["has_personal_data"] is False
    assert ctx["personal_weight"] == 0.0


def test_personal_data_activates_after_3_readings(tracker):
    for i in range(3):
        tracker.update({"CRP": 2.0, "ESR": 10})
    ctx = tracker.get_personal_context({"CRP": 2.0, "ESR": 10})
    assert ctx["has_personal_data"] is True


def test_personal_weight_ramps_with_readings():
    tracker = BiomarkerBaselineTracker(
        biomarkers=["CRP"], personalization_days=10,
    )
    for i in range(3):
        tracker.update({"CRP": 2.0})
    ctx = tracker.get_personal_context({"CRP": 2.0})
    # After 3 readings, weight should be 3/10 = 0.3
    assert abs(ctx["personal_weight"] - 0.3) < 1e-9


def test_personal_weight_caps_at_0_8():
    tracker = BiomarkerBaselineTracker(
        biomarkers=["CRP"], personalization_days=10,
    )
    for i in range(50):
        tracker.update({"CRP": 2.0})
    ctx = tracker.get_personal_context({"CRP": 2.0})
    assert ctx["personal_weight"] == 0.8


def test_normal_reading_interpreted_as_normal(tracker):
    """A reading near the baseline (within natural variance) should be NORMAL."""
    # Provide varied readings so IQR is meaningful (not zero)
    import random
    random.seed(0)
    for _ in range(10):
        tracker.update({
            "CRP": 2.0 + random.uniform(-0.5, 0.5),
            "ESR": 10 + random.uniform(-1.5, 1.5),
        })
    ctx = tracker.get_personal_context({"CRP": 2.1, "ESR": 10.5})
    assert ctx["biomarkers"]["CRP"]["interpretation"] == "NORMAL"


def test_critical_reading_interpreted_as_critical(tracker):
    """Very large deviation (5+ IQRs above) should trigger CRITICAL."""
    import random
    random.seed(0)
    for _ in range(10):
        tracker.update({
            "CRP": 2.0 + random.uniform(-0.5, 0.5),
            "ESR": 10 + random.uniform(-1.5, 1.5),
        })
    # Spike massively to make sure anomaly_score > 3
    ctx = tracker.get_personal_context({"CRP": 50.0, "ESR": 10})
    crp_ctx = ctx["biomarkers"]["CRP"]
    assert crp_ctx["anomaly_score"] > 3.0
    assert crp_ctx["interpretation"] == "CRITICAL"


def test_trend_rising_detected():
    """An obvious rising trajectory should produce RISING trend."""
    import random
    random.seed(0)
    tracker = BiomarkerBaselineTracker(biomarkers=["CRP"])
    # Steady baseline with small variance
    for _ in range(10):
        tracker.update({"CRP": 2.0 + random.uniform(-0.3, 0.3)})
    # Then a clear ramp
    for value in [3.0, 5.0, 7.0, 10.0, 15.0]:
        tracker.update({"CRP": value})
    ctx = tracker.get_personal_context({"CRP": 18.0})
    assert ctx["biomarkers"]["CRP"]["trend"] == "RISING"


def test_outliers_excluded_from_clean_history():
    """A clearly outlier reading should be excluded from clean history."""
    import random
    random.seed(0)
    tracker = BiomarkerBaselineTracker(biomarkers=["CRP"])
    # 10 stable readings with small variance
    for _ in range(10):
        tracker.update({"CRP": 2.0 + random.uniform(-0.3, 0.3)})
    # One huge outlier
    tracker.update({"CRP": 100.0})
    # Verify it was tracked as an outlier
    ctx = tracker.get_personal_context({"CRP": 2.0})
    assert ctx["biomarkers"]["CRP"]["outliers_excluded"] >= 1


def test_baseline_resistant_to_one_outlier():
    """Median + IQR baseline must not move much after one outlier."""
    import random
    random.seed(0)
    tracker = BiomarkerBaselineTracker(biomarkers=["CRP"])
    for _ in range(10):
        tracker.update({"CRP": 2.0 + random.uniform(-0.3, 0.3)})
    # Get baseline before outlier
    ctx1 = tracker.get_personal_context({"CRP": 2.0})
    baseline_before = ctx1["biomarkers"]["CRP"]["median_baseline"]

    # Inject outlier
    tracker.update({"CRP": 100.0})

    ctx2 = tracker.get_personal_context({"CRP": 2.0})
    baseline_after = ctx2["biomarkers"]["CRP"]["median_baseline"]

    # Baseline should not have moved much (the outlier should have been excluded)
    assert abs(baseline_after - baseline_before) < 0.5


def test_missing_biomarker_value_skipped():
    """If a biomarker is missing from the reading, it should be skipped."""
    tracker = BiomarkerBaselineTracker(biomarkers=["CRP", "ESR"])
    for _ in range(5):
        tracker.update({"CRP": 2.0, "ESR": 10})
    ctx = tracker.get_personal_context({"CRP": 2.0})  # ESR missing
    assert "CRP" in ctx["biomarkers"]
    assert "ESR" not in ctx["biomarkers"]


def test_clean_history_exposed():
    """Tracker should expose clean_history for downstream use."""
    tracker = BiomarkerBaselineTracker(biomarkers=["CRP"])
    for _ in range(5):
        tracker.update({"CRP": 2.0})
    assert "CRP" in tracker.clean_history
    assert len(tracker.clean_history["CRP"]) == 5
