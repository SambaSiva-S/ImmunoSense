"""Layer 3 personal baseline trackers for Agent 3 (Environment).

Five parallel univariate trackers (one per environmental feature), wrapped
by a multi-feature container. Same pattern as Agents 2 and 5, but tailored
to environmental features.

Differs from the shared ``RobustBaselineTracker`` in ``agents.common.trackers``
because the environment agent works with ``DailyEnvironmentSummary`` objects
rather than generic feature dicts.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


# Canonical environmental feature names tracked by Layer 3
ENV_FEATURES = [
    "pm25_ug_m3",
    "ozone_ppb",
    "uv_index",
    "barometric_change_kpa",
    "pollen_index",
]


class _UnivariateRobustTracker:
    """Rolling median + IQR with outlier exclusion for one feature.

    Activates after 3 clean readings. Personal weight ramps to 1.0 over
    ``personalization_days`` (default 25).

    Args:
        window: Rolling window size (number of clean readings kept).
        anomaly_threshold: Outlier threshold in IQR units.
        personalization_days: Days of clean history at which personal_weight = 1.0.
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

    def update(self, value: Optional[float]) -> None:
        """Append a value, applying outlier filtering for clean history."""
        if value is None or pd.isna(value):
            return
        self.all_history.append(float(value))

        if len(self.clean_history) < 3:
            self.clean_history.append(float(value))
            return

        arr = np.array(self.clean_history)
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            self.clean_history.append(float(value))
        elif abs(value - med) / iqr < self.anomaly_threshold:
            self.clean_history.append(float(value))
        # else: outlier, exclude from clean history (kept in all_history)

    def baseline(self) -> dict:
        """Return baseline stats: median, IQR, n_clean, personal_weight."""
        n_clean = len(self.clean_history)
        if n_clean < 3:
            return {
                "median": float("nan"),
                "iqr": float("nan"),
                "n_clean": n_clean,
                "personal_weight": 0.0,
            }
        arr = np.array(self.clean_history)
        med = float(np.median(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        return {
            "median": med,
            "iqr": float(q75 - q25),
            "n_clean": n_clean,
            "personal_weight": min(1.0, n_clean / self.personalization_days),
        }

    def anomaly_score(self, value: Optional[float]) -> float:
        """Compute anomaly score (value - median) / IQR. Returns NaN if cold start."""
        if value is None or pd.isna(value) or len(self.clean_history) < 3:
            return float("nan")
        arr = np.array(self.clean_history)
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            return 0.0
        return float((value - med) / iqr)


class EnvironmentRobustTracker:
    """Per-patient baselines for the 5 environmental features.

    Args:
        window: Rolling window for each univariate tracker.
        anomaly_threshold: Outlier threshold in IQR units.
        personalization_days: Days until full personal weight.
    """

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
    ) -> None:
        self.trackers: dict[str, _UnivariateRobustTracker] = {
            f: _UnivariateRobustTracker(window, anomaly_threshold, personalization_days)
            for f in ENV_FEATURES
        }

    def update(self, daily_summary) -> None:
        """Update all 5 trackers from a DailyEnvironmentSummary."""
        for f in ENV_FEATURES:
            v = getattr(daily_summary, f, None)
            if v is not None:
                self.trackers[f].update(v)

    def report(self) -> dict:
        """Return per-feature baseline stats."""
        return {f: t.baseline() for f, t in self.trackers.items()}

    def anomaly_scores(self, daily_summary) -> dict:
        """Return per-feature anomaly scores for one daily summary."""
        return {
            f: self.trackers[f].anomaly_score(getattr(daily_summary, f, None))
            for f in ENV_FEATURES
        }
