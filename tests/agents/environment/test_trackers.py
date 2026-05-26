"""Tests for environment.trackers - per-feature baseline tracking."""

from __future__ import annotations

import numpy as np
import pytest

from immunosense.agents.environment.trackers import (
    ENV_FEATURES,
    EnvironmentRobustTracker,
    _UnivariateRobustTracker,
)
from immunosense.agents.environment.types import DailyEnvironmentSummary


class TestUnivariateRobustTracker:
    def test_cold_start_no_baseline(self):
        t = _UnivariateRobustTracker()
        b = t.baseline()
        assert np.isnan(b["median"])
        assert b["n_clean"] == 0
        assert b["personal_weight"] == 0.0

    def test_after_3_readings_baseline_active(self):
        t = _UnivariateRobustTracker()
        for v in [10.0, 11.0, 9.0]:
            t.update(v)
        b = t.baseline()
        assert b["n_clean"] == 3
        assert b["median"] == 10.0
        assert b["personal_weight"] > 0

    def test_outlier_excluded_from_clean(self):
        t = _UnivariateRobustTracker(anomaly_threshold=2.0)
        for v in [10.0, 11.0, 9.0, 10.5, 9.5]:
            t.update(v)

        # Inject obvious outlier
        t.update(100.0)

        # Clean should not include the outlier; all_history should
        assert 100.0 not in t.clean_history
        assert 100.0 in t.all_history

    def test_anomaly_score_correct(self):
        t = _UnivariateRobustTracker()
        for v in [10.0, 11.0, 9.0, 10.5, 9.5, 10.0, 11.0]:
            t.update(v)
        # Reading at median should have anomaly_score ~ 0
        score = t.anomaly_score(10.0)
        assert abs(score) < 0.5
        # Reading well above should be positive
        score = t.anomaly_score(20.0)
        assert score > 2.0

    def test_anomaly_score_cold_start_returns_nan(self):
        t = _UnivariateRobustTracker()
        t.update(10.0)
        assert np.isnan(t.anomaly_score(20.0))

    def test_none_values_ignored(self):
        t = _UnivariateRobustTracker()
        t.update(None)
        t.update(np.nan)
        assert len(t.all_history) == 0

    def test_personalization_weight_ramps(self):
        t = _UnivariateRobustTracker(personalization_days=10)
        for _ in range(10):
            t.update(10.0)
        b = t.baseline()
        # Note: window defaults to 14, so 10 readings all stored. n_clean might be <= 10.
        assert b["personal_weight"] == pytest.approx(min(1.0, b["n_clean"] / 10), abs=0.05)


class TestEnvironmentRobustTracker:
    def _make_summary(
        self,
        date="2026-04-01",
        pm25=10.0,
        ozone=50.0,
        uv=6.0,
        barometric=0.4,
        pollen=4.0,
    ):
        return DailyEnvironmentSummary(
            date=date,
            location={"region": "SE", "season": "spring"},
            pm25_ug_m3=pm25,
            ozone_ppb=ozone,
            uv_index=uv,
            barometric_change_kpa=barometric,
            pollen_index=pollen,
        )

    def test_all_features_tracked(self):
        tracker = EnvironmentRobustTracker()
        assert set(tracker.trackers.keys()) == set(ENV_FEATURES)

    def test_update_routes_to_per_feature_trackers(self):
        tracker = EnvironmentRobustTracker()
        s = self._make_summary(pm25=15.0, ozone=70.0)
        tracker.update(s)

        assert 15.0 in tracker.trackers["pm25_ug_m3"].all_history
        assert 70.0 in tracker.trackers["ozone_ppb"].all_history

    def test_report_has_all_features(self):
        tracker = EnvironmentRobustTracker()
        for i in range(5):
            tracker.update(self._make_summary())
        report = tracker.report()
        assert set(report.keys()) == set(ENV_FEATURES)
        for f in ENV_FEATURES:
            assert "median" in report[f]
            assert "n_clean" in report[f]

    def test_anomaly_scores_for_one_summary(self):
        tracker = EnvironmentRobustTracker()
        # Diverse baseline so IQR is non-zero
        baseline_values = [
            (8.0, 45.0, 5.5, 0.3, 4.0),
            (9.5, 50.0, 6.0, 0.4, 4.5),
            (10.0, 48.0, 5.8, 0.5, 5.0),
            (11.0, 52.0, 6.2, 0.3, 4.2),
            (9.0, 47.0, 5.5, 0.4, 4.8),
            (10.5, 49.0, 5.9, 0.4, 4.3),
        ]
        for i, (pm, oz, uv, bar, pol) in enumerate(baseline_values):
            tracker.update(self._make_summary(
                date=f"2026-04-{i+1:02d}",
                pm25=pm, ozone=oz, uv=uv, barometric=bar, pollen=pol,
            ))

        # PM2.5 = 30 is way outside baseline of ~10
        scores = tracker.anomaly_scores(self._make_summary(pm25=30.0))

        # Should yield high anomaly for PM2.5
        assert scores["pm25_ug_m3"] > 1.0
        # Others near baseline should be near 0
        assert abs(scores["ozone_ppb"]) < 1.0

    def test_none_value_skipped(self):
        tracker = EnvironmentRobustTracker()
        s = self._make_summary(pm25=None)
        tracker.update(s)
        # PM25 tracker shouldn't have received None
        assert len(tracker.trackers["pm25_ug_m3"].all_history) == 0
        # Others received their values
        assert len(tracker.trackers["ozone_ppb"].all_history) == 1
