"""BiomarkerBaselineTracker - per-patient biomarker tracking.

Wraps the shared RobustBaselineTracker from agents.common with biomarker-
specific context interpretation (CRITICAL / ELEVATED / SUPPRESSED / NORMAL
thresholds and RISING / FALLING / STABLE trend detection).

The shared tracker handles median + IQR + outlier exclusion; this class adds
the biomarker semantics on top.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from immunosense.agents.biomarker.constants import (
    BIOMARKERS_FOR_TRACKING,
    LAYER3_HYPERPARAMS,
)
from immunosense.agents.common.trackers import RobustBaselineTracker


class BiomarkerBaselineTracker:
    """Per-biomarker robust baseline tracker with semantic interpretation.

    Args:
        biomarkers: list of biomarker names to track (default BIOMARKERS_FOR_TRACKING)
        window: rolling window for median/IQR (default 10)
        outlier_threshold: anomaly_score above this excludes the reading from
            clean history (default 2.0 IQRs)
        personalization_days: number of clean readings before personal_weight = 0.8
    """

    def __init__(
        self,
        biomarkers: Optional[list] = None,
        window: int = None,
        outlier_threshold: float = None,
        personalization_days: int = None,
    ) -> None:
        self.biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)
        self.window = window or LAYER3_HYPERPARAMS["tracker_window"]
        self.outlier_threshold = (
            outlier_threshold if outlier_threshold is not None
            else LAYER3_HYPERPARAMS["tracker_outlier_threshold"]
        )
        self.personalization_days = (
            personalization_days or LAYER3_HYPERPARAMS["tracker_personalization_days"]
        )

        # Delegate to the shared multi-feature tracker
        self._tracker = RobustBaselineTracker(
            features=self.biomarkers,
            window=self.window,
            outlier_threshold=self.outlier_threshold,
        )
        self.n_readings = 0

    def update(self, reading: dict) -> None:
        """Update tracker state with one biomarker reading.

        Args:
            reading: dict mapping biomarker name -> value (None/NaN handled).
        """
        self.n_readings += 1
        # The shared tracker accepts a per-feature dict
        self._tracker.update(reading)

    def get_personal_context(self, reading: dict) -> dict:
        """Compare a reading against the personal baseline.

        Returns a dict with structure:
            {
                "has_personal_data": bool,
                "readings_count": int,
                "personal_weight": float in [0.0, 0.8],
                "biomarkers": {
                    biomarker_name: {
                        "value": float,
                        "median_baseline": float,
                        "iqr": float,
                        "q25": float, "q75": float,
                        "anomaly_score": float (in IQRs),
                        "z_score": float (in stds),
                        "ratio": float,
                        "trend": "RISING" | "FALLING" | "STABLE" | "UNKNOWN",
                        "interpretation": "CRITICAL" | "ELEVATED" | "SUPPRESSED"
                                          | "MILDLY_ELEVATED" | "MILDLY_LOW" | "NORMAL",
                        "outliers_excluded": int,
                    }
                }
            }
        """
        if self.n_readings < 3:
            return {
                "has_personal_data": False,
                "readings_count": self.n_readings,
                "personal_weight": 0.0,
                "biomarkers": {},
            }

        # Personal weight: cold-start ramp 0 -> 0.8 over personalization_days readings
        personal_weight = min(0.8, self.n_readings / self.personalization_days)

        context = {
            "has_personal_data": True,
            "readings_count": self.n_readings,
            "personal_weight": personal_weight,
            "biomarkers": {},
        }

        for bm in self.biomarkers:
            if bm not in reading or reading[bm] is None:
                continue
            value = reading[bm]
            try:
                if np.isnan(value):
                    continue
            except (TypeError, ValueError):
                continue

            clean = self._tracker.clean_history.get(bm, [])
            if len(clean) < 3:
                continue

            recent_clean = clean[-self.window:]
            median = float(np.median(recent_clean))
            q25 = float(np.percentile(recent_clean, 25))
            q75 = float(np.percentile(recent_clean, 75))
            iqr = max(q75 - q25, 0.01)
            std = float(np.std(recent_clean))

            anomaly_score = (value - median) / iqr if iqr > 0 else 0.0
            z_score = (value - median) / std if std > 0 else 0.0
            ratio = value / median if median > 0 else 0.0

            # Trend from ALL history (including outliers — full picture)
            all_recent = self._tracker.history.get(bm, [])[-5:]
            if len(all_recent) >= 3:
                slope = float(np.polyfit(range(len(all_recent)), all_recent, 1)[0])
                if slope > iqr * 0.05:
                    trend = "RISING"
                elif slope < -iqr * 0.05:
                    trend = "FALLING"
                else:
                    trend = "STABLE"
            else:
                trend = "UNKNOWN"

            # Semantic interpretation based on anomaly score (IQR-based, robust)
            if anomaly_score > 3:
                interpretation = "CRITICAL"
            elif anomaly_score > 2:
                interpretation = "ELEVATED" if value > median else "SUPPRESSED"
            elif anomaly_score > 1.2:
                interpretation = "MILDLY_ELEVATED" if value > median else "MILDLY_LOW"
            else:
                interpretation = "NORMAL"

            context["biomarkers"][bm] = {
                "value": round(float(value), 2),
                "median_baseline": round(median, 2),
                "iqr": round(iqr, 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "anomaly_score": round(anomaly_score, 2),
                "z_score": round(z_score, 2),
                "ratio": round(ratio, 2),
                "trend": trend,
                "interpretation": interpretation,
                "outliers_excluded": int(
                    self._tracker.n_outliers_detected.get(bm, 0)
                ),
            }

        return context

    @property
    def clean_history(self) -> dict:
        """Expose clean history per biomarker (for plotting / debugging)."""
        return self._tracker.clean_history

    @property
    def history(self) -> dict:
        """Expose all-history per biomarker (including outliers)."""
        return self._tracker.history
