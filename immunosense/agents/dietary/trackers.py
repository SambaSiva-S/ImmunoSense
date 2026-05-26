"""Layer 3 trackers for Agent 2.

OvernightFastTracker
    Computes today's overnight fast hours from yesterday's last_meal_timestamp
    and today's first_meal_timestamp. Stateful across days.

DietaryRobustTracker
    Per-feature personal baselines (median + IQR with outlier exclusion) for the
    6 continuous features, plus boolean prevalence tracking for the 4 trigger
    booleans. Wraps the shared _UnivariateRobustTracker primitive.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.dietary.constants import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
)


class OvernightFastTracker:
    """Computes overnight fast hours (autophagy / mTOR proxy) from timestamps.

    Today's overnight fast =
        today.first_meal_timestamp - yesterday.last_meal_timestamp

    Returns NaN for the first observation (no yesterday yet), if either
    timestamp is missing, or if the computed gap is negative / > 48 hours
    (indicates bad timestamp data).
    """

    def __init__(self) -> None:
        self._previous_last_meal_ts: Optional[str] = None

    def compute(self, rollup) -> float:
        """Update internal state from rollup and return today's overnight fast.

        Args:
            rollup: Anything with .first_meal_timestamp and .last_meal_timestamp
                     attributes (str or None).

        Returns:
            Overnight fast in hours, or float('nan') if not computable.
        """
        if rollup.first_meal_timestamp is None or self._previous_last_meal_ts is None:
            # No today's first meal, OR first day (no yesterday yet)
            self._previous_last_meal_ts = rollup.last_meal_timestamp
            return float("nan")

        try:
            t_first = pd.Timestamp(rollup.first_meal_timestamp)
            t_prev_last = pd.Timestamp(self._previous_last_meal_ts)
        except Exception:
            self._previous_last_meal_ts = rollup.last_meal_timestamp
            return float("nan")

        gap_hours = (t_first - t_prev_last).total_seconds() / 3600.0
        self._previous_last_meal_ts = rollup.last_meal_timestamp

        # Sanity-check: gap must be in a plausible range
        if gap_hours < 0 or gap_hours > 48:
            return float("nan")
        return float(gap_hours)


class _UnivariateRobustTracker:
    """Rolling median + IQR with outlier exclusion.

    Same algorithmic pattern as the shared RobustBaselineTracker in
    agents.common.trackers, but as a single-feature primitive for use
    inside DietaryRobustTracker's feature-keyed container.

    Args:
        window: Maximum number of observations to keep in history.
        anomaly_threshold: Observations more than this many IQRs from the
                            current median are excluded from clean_history
                            (treated as outliers, kept in all_history).
        personalization_days: Number of clean observations needed to reach
                              full personal_weight=1.0.
    """

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
    ) -> None:
        self.window = window
        self.anomaly_threshold = anomaly_threshold
        self.personalization_days = personalization_days
        self.all_history: deque = deque(maxlen=window)
        self.clean_history: deque = deque(maxlen=window)

    def update(self, value: float) -> None:
        """Add an observation. Outliers go to all_history only."""
        if value is None or pd.isna(value):
            return
        self.all_history.append(float(value))

        # Need at least 3 observations to compute IQR for outlier detection
        if len(self.clean_history) < 3:
            self.clean_history.append(float(value))
            return

        arr = np.array(self.clean_history)
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            # Zero variance — accept new observation
            self.clean_history.append(float(value))
            return

        deviation = abs(value - med) / iqr
        if deviation < self.anomaly_threshold:
            self.clean_history.append(float(value))
        # else: silently exclude — this is an outlier vs. current baseline

    def baseline(self) -> dict:
        """Current personal baseline stats."""
        n_all = len(self.all_history)
        n_clean = len(self.clean_history)

        if n_clean < 3:
            return {
                "median": float("nan"),
                "iqr": float("nan"),
                "n_clean": n_clean,
                "n_all": n_all,
                "personal_weight": 0.0,
            }

        arr = np.array(self.clean_history)
        med = float(np.median(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float(q75 - q25)
        personal_weight = min(1.0, n_clean / self.personalization_days)

        return {
            "median": med,
            "iqr": iqr,
            "n_clean": n_clean,
            "n_all": n_all,
            "personal_weight": personal_weight,
        }

    def anomaly_score(self, value: float) -> float:
        """Z-like score using personal median and IQR.

        Returns:
            (value - median) / IQR. NaN if value missing or fewer than
            3 clean observations. 0.0 if IQR is zero (degenerate baseline).
        """
        if pd.isna(value) or len(self.clean_history) < 3:
            return float("nan")
        arr = np.array(self.clean_history)
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            return 0.0
        return float((value - med) / iqr)


class DietaryRobustTracker:
    """Per-patient baseline machinery for the 10 daily dietary features.

    Tracks 6 continuous features with robust median+IQR baselines
    (via _UnivariateRobustTracker), plus 4 boolean triggers as prevalence.
    """

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
    ) -> None:
        self.continuous_trackers = {
            f: _UnivariateRobustTracker(window, anomaly_threshold, personalization_days)
            for f in CONTINUOUS_FEATURES
        }
        self.boolean_history = {f: deque(maxlen=window) for f in BOOLEAN_TRIGGERS}
        self.window = window

    def update(self, daily_features: dict) -> None:
        """Update all per-feature trackers from one day's feature dict."""
        for f in CONTINUOUS_FEATURES:
            v = daily_features.get(f)
            if v is not None:
                self.continuous_trackers[f].update(v)
        for f in BOOLEAN_TRIGGERS:
            v = daily_features.get(f)
            if v is not None:
                self.boolean_history[f].append(bool(v))

    def report(self) -> dict:
        """Return current baselines for all features."""
        out = {"continuous": {}, "boolean": {}}
        for f, t in self.continuous_trackers.items():
            out["continuous"][f] = t.baseline()
        for f, h in self.boolean_history.items():
            if len(h) == 0:
                out["boolean"][f] = {"prevalence": float("nan"), "n_days": 0}
            else:
                out["boolean"][f] = {
                    "prevalence": float(np.mean(h)),
                    "n_days": len(h),
                }
        return out

    def anomaly_scores(self, daily_features: dict) -> dict:
        """Per-feature anomaly score for the given day's values."""
        return {
            f: self.continuous_trackers[f].anomaly_score(
                daily_features.get(f, float("nan"))
            )
            for f in CONTINUOUS_FEATURES
        }
