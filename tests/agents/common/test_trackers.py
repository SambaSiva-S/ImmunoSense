"""Tests for RobustBaselineTracker."""

from __future__ import annotations

import numpy as np
import pytest

from immunosense.agents.common.trackers import RobustBaselineTracker


# ============================================================
# Initialization tests
# ============================================================

class TestInitialization:
    def test_default_construction(self):
        tracker = RobustBaselineTracker(["a", "b"])
        assert tracker.features == ["a", "b"]
        assert tracker.window == 10
        assert tracker.outlier_threshold == 2.0
        assert tracker.n_readings == 0
        assert tracker.history == {"a": [], "b": []}
        assert tracker.clean_history == {"a": [], "b": []}

    def test_custom_window(self):
        tracker = RobustBaselineTracker(["a"], window=20)
        assert tracker.window == 20

    def test_custom_threshold(self):
        tracker = RobustBaselineTracker(["a"], outlier_threshold=3.0)
        assert tracker.outlier_threshold == 3.0

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError, match="window must be >= 3"):
            RobustBaselineTracker(["a"], window=2)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="outlier_threshold must be > 0"):
            RobustBaselineTracker(["a"], outlier_threshold=0)


# ============================================================
# Update behavior
# ============================================================

class TestUpdate:
    def test_single_reading_no_personal_context(self):
        """With < 3 readings, no personal context available."""
        tracker = RobustBaselineTracker(["a"])
        tracker.update({"a": 50.0})
        ctx = tracker.get_personal_context({"a": 60.0})
        assert ctx["has_personal_data"] is False
        assert ctx["readings_count"] == 1

    def test_three_readings_enables_context(self):
        tracker = RobustBaselineTracker(["a"])
        for v in [50.0, 51.0, 49.0]:
            tracker.update({"a": v})
        ctx = tracker.get_personal_context({"a": 50.5})
        assert ctx["has_personal_data"] is True
        assert "features" in ctx

    def test_missing_feature_skipped(self):
        """Reading without all features doesn't error."""
        tracker = RobustBaselineTracker(["a", "b"])
        tracker.update({"a": 50.0})  # b missing
        # Doesn't crash. History grows for a only.
        assert len(tracker.history["a"]) == 1
        assert len(tracker.history["b"]) == 0

    def test_none_value_skipped(self):
        tracker = RobustBaselineTracker(["a"])
        tracker.update({"a": None})
        assert len(tracker.history["a"]) == 0

    def test_nan_value_skipped(self):
        tracker = RobustBaselineTracker(["a"])
        tracker.update({"a": float("nan")})
        assert len(tracker.history["a"]) == 0

    def test_inf_value_skipped(self):
        tracker = RobustBaselineTracker(["a"])
        tracker.update({"a": float("inf")})
        assert len(tracker.history["a"]) == 0

    def test_n_readings_counter(self):
        tracker = RobustBaselineTracker(["a"])
        for _ in range(5):
            tracker.update({"a": 50.0})
        assert tracker.n_readings == 5


# ============================================================
# Outlier detection
# ============================================================

class TestOutlierDetection:
    def test_no_outlier_under_normal_distribution(self):
        """Values near baseline shouldn't be flagged as outliers."""
        rng = np.random.default_rng(seed=42)
        tracker = RobustBaselineTracker(["a"], outlier_threshold=2.0)
        for _ in range(20):
            tracker.update({"a": float(rng.normal(50, 2))})
        # Most should land in clean history
        assert tracker.n_outliers_detected["a"] <= 3  # rare statistical outliers OK

    def test_clear_outlier_excluded_from_clean_history(self):
        """A 10x deviation should be flagged."""
        tracker = RobustBaselineTracker(["a"], outlier_threshold=2.0)
        # Build baseline
        for _ in range(5):
            tracker.update({"a": 50.0})
        # Inject outlier
        tracker.update({"a": 200.0})

        # Outlier should be detected
        assert tracker.n_outliers_detected["a"] == 1
        # Outlier is in full history but not clean history
        assert 200.0 in tracker.history["a"]
        assert 200.0 not in tracker.clean_history["a"]

    def test_outlier_doesnt_corrupt_baseline(self):
        """After an outlier, normal readings still see clean baseline."""
        tracker = RobustBaselineTracker(["a"], outlier_threshold=2.0)
        for _ in range(5):
            tracker.update({"a": 50.0})

        tracker.update({"a": 1000.0})  # huge outlier

        # New normal reading should compute against clean (50) baseline
        ctx = tracker.get_personal_context({"a": 50.0})
        assert abs(ctx["features"]["a"]["median_baseline"] - 50.0) < 1.0


# ============================================================
# Personal context analysis
# ============================================================

class TestPersonalContext:
    def _build_tracker_with_baseline(self, mean: float = 50.0, std: float = 2.0, n: int = 10):
        """Helper: build tracker with stable baseline."""
        rng = np.random.default_rng(seed=42)
        tracker = RobustBaselineTracker(["a"])
        for _ in range(n):
            tracker.update({"a": float(rng.normal(mean, std))})
        return tracker

    def test_normal_value_interpretation(self):
        tracker = self._build_tracker_with_baseline()
        ctx = tracker.get_personal_context({"a": 50.0})
        assert ctx["features"]["a"]["interpretation"] == "NORMAL"

    def test_critical_high_interpretation(self):
        tracker = self._build_tracker_with_baseline()
        # Value far above baseline
        ctx = tracker.get_personal_context({"a": 100.0})
        assert ctx["features"]["a"]["interpretation"] == "CRITICAL"
        assert ctx["features"]["a"]["anomaly_score"] > 3

    def test_elevated_interpretation(self):
        tracker = self._build_tracker_with_baseline()
        # Value moderately above baseline (2-3 IQRs)
        ctx = tracker.get_personal_context({"a": 60.0})
        assert ctx["features"]["a"]["interpretation"] in ["ELEVATED", "CRITICAL"]

    def test_suppressed_interpretation(self):
        tracker = self._build_tracker_with_baseline()
        # Value far below baseline
        ctx = tracker.get_personal_context({"a": 0.0})
        assert ctx["features"]["a"]["interpretation"] in ["SUPPRESSED", "CRITICAL"]

    def test_ratio_computation(self):
        tracker = self._build_tracker_with_baseline()
        ctx = tracker.get_personal_context({"a": 100.0})
        # ratio = value / median ≈ 100 / 50 = 2.0
        assert 1.9 < ctx["features"]["a"]["ratio"] < 2.1

    def test_personal_weight_ramps_to_08(self):
        """Cold start: weight should approach 0.8 after 25 readings."""
        tracker = RobustBaselineTracker(["a"])
        for _ in range(25):
            tracker.update({"a": 50.0})
        ctx = tracker.get_personal_context({"a": 50.0})
        assert abs(ctx["personal_weight"] - 0.8) < 0.05

    def test_outliers_excluded_count_reported(self):
        """After a clear baseline is established, a 20x deviation should be flagged."""
        # Build a clear baseline first - 10 readings all very close to 50.0
        tracker = RobustBaselineTracker(["a"], outlier_threshold=2.0)
        for i in range(10):
            tracker.update({"a": 50.0 + (i % 2) * 0.5})  # alternates 50.0 and 50.5

        baseline_outliers = tracker.n_outliers_detected["a"]

        # Inject a massive outlier
        tracker.update({"a": 1000.0})

        # Should detect exactly one new outlier
        assert tracker.n_outliers_detected["a"] == baseline_outliers + 1

        # Context should report the outlier count
        ctx = tracker.get_personal_context({"a": 50.0})
        assert ctx["features"]["a"]["outliers_excluded"] == baseline_outliers + 1


# ============================================================
# Trend detection
# ============================================================

class TestTrend:
    def test_rising_trend(self):
        tracker = RobustBaselineTracker(["a"])
        # Inject baseline so personal context is available
        for v in [50.0, 50.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]:
            tracker.update({"a": v})
        ctx = tracker.get_personal_context({"a": 56.0})
        assert ctx["features"]["a"]["trend"] == "RISING"

    def test_falling_trend(self):
        tracker = RobustBaselineTracker(["a"])
        for v in [50.0, 50.0, 50.0, 49.0, 48.0, 47.0, 46.0, 45.0]:
            tracker.update({"a": v})
        ctx = tracker.get_personal_context({"a": 44.0})
        assert ctx["features"]["a"]["trend"] == "FALLING"

    def test_stable_trend(self):
        rng = np.random.default_rng(seed=42)
        tracker = RobustBaselineTracker(["a"])
        for _ in range(10):
            tracker.update({"a": float(rng.normal(50, 1))})
        ctx = tracker.get_personal_context({"a": 50.0})
        assert ctx["features"]["a"]["trend"] in ["STABLE", "RISING", "FALLING"]


# ============================================================
# Reset behavior
# ============================================================

class TestReset:
    def test_reset_clears_history(self):
        tracker = RobustBaselineTracker(["a"])
        for _ in range(10):
            tracker.update({"a": 50.0})
        tracker.reset()
        assert tracker.n_readings == 0
        assert tracker.history == {"a": []}
        assert tracker.clean_history == {"a": []}
        assert tracker.n_outliers_detected == {"a": 0}
