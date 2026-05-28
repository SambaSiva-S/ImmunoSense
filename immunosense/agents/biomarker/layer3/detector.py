"""PatternDetector - find trigger -> biomarker correlations across time lags.

For each (trigger, biomarker, lag) combination, computes the Pearson
correlation between the trigger value at time t and the biomarker value at
time t+lag. Patterns with |r| > threshold are reported.

Also derives a per-biomarker flare rule (mean during flare vs. mean during
normal periods).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from immunosense.agents.biomarker.constants import (
    BIOMARKERS_FOR_TRACKING,
    BIOMARKER_TRIGGERS,
    LAYER3_HYPERPARAMS,
)


class PatternDetector:
    """Detects trigger -> biomarker correlations across time lags.

    Args:
        biomarkers: biomarker names to test (default BIOMARKERS_FOR_TRACKING)
        triggers: trigger names to test (default BIOMARKER_TRIGGERS)
        lag_range: tuple (min_lag, max_lag) inclusive — number of readings ahead
        min_readings: minimum trajectory length before patterns can be detected
        correlation_threshold: |r| above which a pattern is reported
        strong_threshold: |r| above which the pattern is marked STRONG
    """

    def __init__(
        self,
        biomarkers: Optional[list] = None,
        triggers: Optional[list] = None,
        lag_range: Optional[tuple] = None,
        min_readings: int = None,
        correlation_threshold: float = None,
        strong_threshold: float = None,
    ) -> None:
        self.biomarkers = list(biomarkers or BIOMARKERS_FOR_TRACKING)
        self.triggers = list(triggers or BIOMARKER_TRIGGERS)
        self.lag_range = lag_range or LAYER3_HYPERPARAMS["detector_lag_range"]
        self.min_readings = (
            min_readings or LAYER3_HYPERPARAMS["detector_min_readings"]
        )
        self.correlation_threshold = (
            correlation_threshold
            if correlation_threshold is not None
            else LAYER3_HYPERPARAMS["detector_correlation_threshold"]
        )
        self.strong_threshold = (
            strong_threshold
            if strong_threshold is not None
            else LAYER3_HYPERPARAMS["detector_strong_correlation"]
        )
        self.detected_patterns: list = []

    def analyze(self, trajectory: list) -> dict:
        """Find correlation patterns in a sequence of readings.

        Args:
            trajectory: list of dicts, each with biomarker + trigger keys.

        Returns:
            {
                "has_patterns": bool,
                "patterns": list[dict] sorted by |correlation| desc,
                "flare_rule": dict | None,
                "n_readings_analyzed": int,
            }
        """
        if len(trajectory) < self.min_readings:
            return {
                "has_patterns": False,
                "message": f"Need {self.min_readings}+ readings",
                "patterns": [],
                "flare_rule": None,
                "n_readings_analyzed": len(trajectory),
            }

        self.detected_patterns = []

        for bm in self.biomarkers:
            bm_values = [r.get(bm) for r in trajectory]
            if any(v is None for v in bm_values):
                continue
            bm_array = np.array(bm_values, dtype=float)
            if np.isnan(bm_array).any():
                continue

            for trigger in self.triggers:
                trig_values = [1 if r.get(trigger, False) else 0 for r in trajectory]
                trig_array = np.array(trig_values)

                for lag in range(self.lag_range[0], self.lag_range[1] + 1):
                    if lag >= len(trig_array):
                        continue

                    # Pair: trigger at t, biomarker at t+lag
                    trig_shifted = trig_array[:-lag] if lag > 0 else trig_array
                    bm_shifted = bm_array[lag:] if lag > 0 else bm_array
                    min_len = min(len(trig_shifted), len(bm_shifted))
                    if min_len < 5:
                        continue
                    trig_shifted = trig_shifted[:min_len]
                    bm_shifted = bm_shifted[:min_len]

                    if trig_shifted.std() == 0 or bm_shifted.std() == 0:
                        continue

                    correlation = float(np.corrcoef(trig_shifted, bm_shifted)[0, 1])

                    # Effect size: mean biomarker when exposed vs. unexposed
                    exposed = bm_shifted[trig_shifted == 1]
                    unexposed = bm_shifted[trig_shifted == 0]
                    if len(exposed) >= 2 and len(unexposed) >= 2:
                        effect_size = float(exposed.mean() - unexposed.mean())
                        effect_pct = (
                            (effect_size / unexposed.mean() * 100)
                            if unexposed.mean() != 0 else 0.0
                        )
                    else:
                        effect_size = 0.0
                        effect_pct = 0.0

                    if abs(correlation) > self.correlation_threshold:
                        self.detected_patterns.append({
                            "trigger": trigger,
                            "biomarker": bm,
                            "lag_readings": lag,
                            "correlation": round(correlation, 3),
                            "effect_size": round(effect_size, 2),
                            "effect_pct": round(effect_pct, 1),
                            "n_exposed": int(trig_shifted.sum()),
                            "strength": (
                                "STRONG" if abs(correlation) > self.strong_threshold
                                else "MODERATE"
                            ),
                        })

        # Sort by |correlation| descending
        self.detected_patterns.sort(
            key=lambda p: abs(p["correlation"]), reverse=True,
        )

        flare_rule = self._derive_flare_rule(trajectory)

        return {
            "has_patterns": len(self.detected_patterns) > 0,
            "patterns": self.detected_patterns,
            "flare_rule": flare_rule,
            "n_readings_analyzed": len(trajectory),
        }

    def _derive_flare_rule(self, trajectory: list) -> Optional[dict]:
        """Compute per-biomarker thresholds separating flare from normal readings.

        Returns:
            dict {biomarker: {flare_mean, normal_mean, ratio, threshold}} or None.
        """
        flare_readings = [r for r in trajectory if r.get("is_flare", False)]
        normal_readings = [r for r in trajectory if not r.get("is_flare", False)]

        if len(flare_readings) < 2 or len(normal_readings) < 2:
            return None

        rules = {}
        for bm in self.biomarkers:
            flare_vals = [r[bm] for r in flare_readings if bm in r and r[bm] is not None]
            normal_vals = [r[bm] for r in normal_readings if bm in r and r[bm] is not None]
            if len(flare_vals) < 2 or len(normal_vals) < 2:
                continue

            flare_mean = float(np.mean(flare_vals))
            normal_mean = float(np.mean(normal_vals))
            if normal_mean > 0:
                ratio = flare_mean / normal_mean
                threshold = normal_mean + (flare_mean - normal_mean) * 0.5
                rules[bm] = {
                    "flare_mean": round(flare_mean, 2),
                    "normal_mean": round(normal_mean, 2),
                    "ratio": round(ratio, 2),
                    "threshold": round(threshold, 2),
                }

        return rules if rules else None
