"""Shared baseline trackers used across agents.

The RobustBaselineTracker uses rolling median + IQR to maintain personal
baselines that resist corruption from flare/anomaly readings. Both Agent 1
(Biomarker) and Agent 4 (Wearable) used near-identical implementations;
this is the consolidated canonical version.

Other agents (2 Dietary, 3 Environment, 5 Symptoms) use agent-specific
multi-feature trackers internally, but may compose primitives from here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class RobustBaselineTracker:
    """Tracks personal baselines using rolling median + IQR.

    Outlier-resistant: flare/anomaly readings do not corrupt the baseline.
    Activates after 3 readings. Cold start weight ramps 0.0 → 0.8 over 25 readings.

    Args:
        features: List of feature names to track.
        window: Rolling window size for median/IQR computation.
        outlier_threshold: Reading is excluded from clean history if its
            anomaly_score (|value - median| / iqr) exceeds this threshold.

    Example:
        >>> tracker = RobustBaselineTracker(["hrv_rmssd", "sleep_efficiency"])
        >>> tracker.update({"hrv_rmssd": 32.5, "sleep_efficiency": 0.85})
        >>> ctx = tracker.get_personal_context({"hrv_rmssd": 24.0, "sleep_efficiency": 0.65})
        >>> ctx["has_personal_data"]
        False  # need 3+ readings first
    """

    def __init__(
        self,
        features: list[str],
        window: int = 10,
        outlier_threshold: float = 2.0,
    ) -> None:
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if outlier_threshold <= 0:
            raise ValueError(f"outlier_threshold must be > 0, got {outlier_threshold}")

        self.features = list(features)
        self.window = window
        self.outlier_threshold = outlier_threshold

        # Full history (includes outliers, used for trend analysis)
        self.history: dict[str, list[float]] = {f: [] for f in features}

        # Clean history (excludes outliers, used for baseline)
        self.clean_history: dict[str, list[float]] = {f: [] for f in features}

        # Counters
        self.n_readings: int = 0
        self.n_outliers_detected: dict[str, int] = {f: 0 for f in features}

    def update(self, reading: dict[str, Any]) -> None:
        """Add a new reading to the tracker.

        Outliers are detected (using current clean history) and excluded from
        future baseline computations, but kept in the full history.

        Args:
            reading: Dict with feature names → values. Missing features
                or None values are skipped without error.
        """
        self.n_readings += 1

        for f in self.features:
            if f not in reading or reading[f] is None:
                continue

            value = reading[f]
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                continue

            self.history[f].append(float(value))

            # Outlier detection (needs at least 3 clean readings)
            if len(self.clean_history[f]) >= 3:
                recent = self.clean_history[f][-self.window:]
                median = float(np.median(recent))
                iqr = self._compute_iqr(recent)

                if iqr > 0:
                    anomaly = abs(value - median) / iqr
                    if anomaly > self.outlier_threshold:
                        self.n_outliers_detected[f] += 1
                        continue  # don't add to clean history

            self.clean_history[f].append(float(value))

    @staticmethod
    def _compute_iqr(values: list[float]) -> float:
        """Compute interquartile range with degenerate-case handling."""
        if len(values) < 4:
            return float(np.std(values)) if len(values) > 1 else 1.0
        return max(
            float(np.percentile(values, 75) - np.percentile(values, 25)),
            0.01,
        )

    def get_personal_context(self, reading: dict[str, Any]) -> dict:
        """Compute personal context for a reading against accumulated baseline.

        Returns a dict with:
            has_personal_data: bool — True if 3+ readings accumulated
            readings_count: int — total readings seen
            personal_weight: float — cold-start ramp 0.0 → 0.8 over 25 readings
            features: dict — per-feature analysis (only if has_personal_data)

        Per-feature analysis includes:
            value, median_baseline, iqr, q25, q75, anomaly_score, z_score,
            ratio (value/median), trend (RISING/FALLING/STABLE),
            interpretation (CRITICAL/ELEVATED/SUPPRESSED/MILDLY_*/NORMAL),
            outliers_excluded.
        """
        if self.n_readings < 3:
            return {
                "has_personal_data": False,
                "readings_count": self.n_readings,
                "personal_weight": 0.0,
            }

        ctx: dict = {
            "has_personal_data": True,
            "readings_count": self.n_readings,
            "personal_weight": min(0.8, self.n_readings / 25),
            "features": {},
        }

        for f in self.features:
            if f not in reading or reading[f] is None:
                continue

            clean = self.clean_history[f]
            if len(clean) < 3:
                continue

            value = float(reading[f])
            recent = clean[-self.window:]

            median = float(np.median(recent))
            iqr = self._compute_iqr(recent)
            q25 = float(np.percentile(recent, 25))
            q75 = float(np.percentile(recent, 75))
            std = float(np.std(recent))

            anomaly_score = (value - median) / iqr if iqr > 0 else 0.0
            z_score = (value - median) / std if std > 0 else 0.0
            ratio = value / median if median > 0 else 0.0

            # Trend from ALL history (we want to see the full picture, including outliers)
            all_recent = self.history[f][-5:]
            if len(all_recent) >= 3:
                slope = float(np.polyfit(range(len(all_recent)), all_recent, 1)[0])
                trend_threshold = iqr * 0.05
                if slope > trend_threshold:
                    trend = "RISING"
                elif slope < -trend_threshold:
                    trend = "FALLING"
                else:
                    trend = "STABLE"
            else:
                trend = "UNKNOWN"

            # Interpretation (uses anomaly_score, more robust than z_score)
            if abs(anomaly_score) > 3:
                interpretation = "CRITICAL"
            elif abs(anomaly_score) > 2:
                interpretation = "ELEVATED" if value > median else "SUPPRESSED"
            elif abs(anomaly_score) > 1.2:
                interpretation = "MILDLY_ELEVATED" if value > median else "MILDLY_LOW"
            else:
                interpretation = "NORMAL"

            ctx["features"][f] = {
                "value": round(value, 3),
                "median_baseline": round(median, 3),
                "iqr": round(iqr, 3),
                "q25": round(q25, 3),
                "q75": round(q75, 3),
                "anomaly_score": round(anomaly_score, 2),
                "z_score": round(z_score, 2),
                "ratio": round(ratio, 3),
                "trend": trend,
                "interpretation": interpretation,
                "outliers_excluded": self.n_outliers_detected[f],
            }

        return ctx

    # === Backwards-compatible property aliases ===
    # Agent 1 used "biomarkers" in the output dict; Agent 4 used "features".
    # The canonical key is "features". Adapters can rename as needed.

    def reset(self) -> None:
        """Reset all tracker state. Useful for testing."""
        self.history = {f: [] for f in self.features}
        self.clean_history = {f: [] for f in self.features}
        self.n_readings = 0
        self.n_outliers_detected = {f: 0 for f in self.features}
