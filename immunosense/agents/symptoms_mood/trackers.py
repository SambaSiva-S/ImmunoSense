"""Layer 3 personal baseline trackers for Agent 5.

Ten parallel univariate trackers (8 symptoms + 2 mood scales), wrapped by a
multi-feature container.

Same outlier-resistant median+IQR pattern as Agents 2 and 3.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.symptoms_mood.types import ALL_FEATURES, DailySymptomMoodSummary


class _UnivariateRobustTracker:
    """Rolling median + IQR with outlier exclusion for one feature."""

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
    ) -> None:
        self.window = window
        self.anomaly_threshold = anomaly_threshold
        self.personalization_days = personalization_days
        self.clean_history: deque = deque(maxlen=window)

    def update(self, value: Optional[float]) -> None:
        """Append value, applying outlier filtering once baseline is established."""
        if value is None or pd.isna(value):
            return

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
        # else: outlier, excluded from clean history

    def baseline(self) -> dict:
        """Return baseline stats: median, IQR, n_clean, personal_weight."""
        n = len(self.clean_history)
        if n < 3:
            return {
                "median": float("nan"),
                "iqr": float("nan"),
                "n_clean": n,
                "personal_weight": 0.0,
            }
        arr = np.array(self.clean_history)
        q75, q25 = np.percentile(arr, [75, 25])
        return {
            "median": float(np.median(arr)),
            "iqr": float(q75 - q25),
            "n_clean": n,
            "personal_weight": min(1.0, n / self.personalization_days),
        }

    def anomaly_score(self, value: Optional[float]) -> float:
        """Compute anomaly score (value - median) / IQR. NaN if cold start."""
        if value is None or pd.isna(value) or len(self.clean_history) < 3:
            return float("nan")
        arr = np.array(self.clean_history)
        med = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-9:
            return 0.0
        return float((value - med) / iqr)


class SymptomsMoodRobustTracker:
    """Per-patient baselines for the 10 symptom + mood features."""

    def __init__(
        self,
        window: int = 14,
        anomaly_threshold: float = 2.0,
        personalization_days: int = 25,
    ) -> None:
        self.trackers: dict[str, _UnivariateRobustTracker] = {
            f: _UnivariateRobustTracker(window, anomaly_threshold, personalization_days)
            for f in ALL_FEATURES
        }

    def update(self, daily_summary: DailySymptomMoodSummary) -> None:
        """Update all 10 trackers from a DailySymptomMoodSummary."""
        for f in ALL_FEATURES:
            v = getattr(daily_summary, f, None)
            if v is not None:
                self.trackers[f].update(v)

    def report(self) -> dict:
        """Return per-feature baseline stats."""
        return {f: t.baseline() for f, t in self.trackers.items()}

    def anomaly_scores(self, daily_summary: DailySymptomMoodSummary) -> dict:
        """Return per-feature anomaly scores for one daily summary."""
        return {
            f: self.trackers[f].anomaly_score(getattr(daily_summary, f, None))
            for f in ALL_FEATURES
        }
