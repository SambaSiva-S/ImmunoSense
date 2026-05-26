"""Lagged correlation pattern detector for Agent 4.

Detects trigger-feature correlations across short time lags (1-3 nights).
Uses |Pearson correlation| > 0.25 as the inclusion threshold.

Note: This is Agent 4's lighter-weight correlation-based detector. It's
different from the BH FDR-corrected detectors in Agents 3 and 5, which run
permutation tests with multiple-comparison correction. Agent 4's detector is
intentionally simpler — its outputs feed into the Conductor for cross-agent
corroboration rather than being treated as authoritative on their own.
"""

from __future__ import annotations

import numpy as np


class PatternDetector:
    """Detects trigger-feature correlations across time lags.

    Args:
        features: Feature names to analyze.
        triggers: Boolean trigger names to test against.
        lag_range: Tuple (min_lag, max_lag) in nights. Default (1, 3).
        min_readings: Minimum number of readings required.
    """

    def __init__(
        self,
        features: list[str],
        triggers: list[str],
        lag_range: tuple = (1, 3),
        min_readings: int = 10,
    ) -> None:
        self.features = features
        self.triggers = triggers
        self.lag_range = lag_range
        self.min_readings = min_readings
        self.detected_patterns: list = []

    def analyze(self, trajectory: list) -> dict:
        """Run correlation analysis over a trajectory of readings.

        Args:
            trajectory: List of reading dicts.

        Returns:
            Dict with has_patterns, patterns, n_readings_analyzed.
        """
        if len(trajectory) < self.min_readings:
            return {
                "has_patterns": False,
                "message": f"Need {self.min_readings}+ readings",
            }
        self.detected_patterns = []

        for f in self.features:
            f_vals = [r.get(f) for r in trajectory]
            if any(v is None for v in f_vals):
                continue
            f_arr = np.array(f_vals)

            for trig in self.triggers:
                t_vals = [1 if r.get(trig, False) else 0 for r in trajectory]
                t_arr = np.array(t_vals)

                for lag in range(self.lag_range[0], self.lag_range[1] + 1):
                    if lag >= len(t_arr):
                        continue
                    t_shifted = t_arr[:-lag] if lag > 0 else t_arr
                    f_shifted = f_arr[lag:] if lag > 0 else f_arr
                    n = min(len(t_shifted), len(f_shifted))
                    if n < 5:
                        continue
                    t_shifted, f_shifted = t_shifted[:n], f_shifted[:n]
                    if t_shifted.std() == 0 or f_shifted.std() == 0:
                        continue
                    corr = float(np.corrcoef(t_shifted, f_shifted)[0, 1])

                    exposed = f_shifted[t_shifted == 1]
                    unexposed = f_shifted[t_shifted == 0]
                    if len(exposed) >= 2 and len(unexposed) >= 2:
                        effect_size = float(exposed.mean() - unexposed.mean())
                        effect_pct = (
                            effect_size / unexposed.mean() * 100
                            if unexposed.mean() != 0 else 0.0
                        )
                    else:
                        effect_size = 0.0
                        effect_pct = 0.0

                    if abs(corr) > 0.25:
                        self.detected_patterns.append({
                            "trigger": trig,
                            "feature": f,
                            "lag_readings": lag,
                            "correlation": round(corr, 3),
                            "effect_size": round(effect_size, 3),
                            "effect_pct": round(effect_pct, 1),
                            "n_exposed": int(t_shifted.sum()),
                            "strength": "STRONG" if abs(corr) > 0.5 else "MODERATE",
                        })

        self.detected_patterns.sort(
            key=lambda x: abs(x["correlation"]), reverse=True
        )
        return {
            "has_patterns": len(self.detected_patterns) > 0,
            "patterns": self.detected_patterns,
            "n_readings_analyzed": len(trajectory),
        }
