"""Environmental trigger detector with BH FDR correction.

Hypothesis structure per patient:
    5 features x 4 lags x 1 binarization pathway (p85) = 20 hypotheses

For each (feature, lag) pair:
    1. Binarize the feature at the 85th percentile (high vs not-high)
    2. Compare mean flare severity between high-exposure and not-high days
    3. One-sided permutation test (n_permutations=500) for raw p-value
    4. Wrong-direction findings get p=1.0 (KEPT in BH testing surface, NOT dropped)
    5. Benjamini-Hochberg FDR correction at alpha=0.10

Confidence tiers from BH-adjusted q-values:
    q < 0.01  -> 'high'
    q < 0.05  -> 'medium'
    q < 0.10  -> 'low'

Patterns with q >= 0.10 are suppressed.

Design choices documented in the original notebook:
    - Linear (Pearson) pathway dropped: real autoimmune triggers are threshold-shaped
    - Only p85 percentile (not p75/p90): balances sensitivity vs noise
    - Wrong-direction findings kept in BH testing surface (dropping breaks multiple-testing)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from immunosense.agents.environment.trackers import ENV_FEATURES
from immunosense.agents.environment.types import DetectedPattern


class EnvironmentTriggerDetector:
    """BH FDR-corrected trigger detector for environmental features.

    Args:
        min_days: Minimum days of observation required before detection runs.
        lags: Tuple of lag days to test (0=same day, 1=next day, ...).
        n_permutations: Number of permutations for null distribution.
        binarize_percentile: Percentile threshold for "high exposure" binarization.
        fdr_target: Target FDR level (default 0.10 = 10% FDR control).
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        min_days: int = 14,
        lags: tuple = (0, 1, 2, 3),
        n_permutations: int = 500,
        binarize_percentile: int = 85,
        fdr_target: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        self.min_days = min_days
        self.lags = lags
        self.n_permutations = n_permutations
        self.binarize_percentile = binarize_percentile
        self.fdr_target = fdr_target
        self.rng = np.random.RandomState(random_seed)
        self.last_n_hypotheses = 0

    def detect(
        self,
        daily_records: list,
        flare_severity_by_date: dict,
    ) -> list:
        """Run BH FDR detection on daily records.

        Args:
            daily_records: List of dicts with 'date' key and feature values.
            flare_severity_by_date: Dict mapping date string -> flare severity float.

        Returns:
            List of DetectedPattern objects (only those with q < fdr_target),
            sorted by absolute effect size descending.
        """
        if len(daily_records) < self.min_days:
            self.last_n_hypotheses = 0
            return []

        records_by_date = {r["date"]: r for r in daily_records}
        sorted_dates = sorted(records_by_date.keys())
        candidates = self._build_candidates(sorted_dates, records_by_date, flare_severity_by_date)

        self.last_n_hypotheses = len(candidates)
        if not candidates:
            return []

        return self._apply_bh_correction(candidates)

    def _build_candidates(
        self,
        sorted_dates: list,
        records_by_date: dict,
        flare_severity_by_date: dict,
    ) -> list:
        """Build candidate hypotheses for each (feature, lag) pair."""
        candidates = []

        for lag in self.lags:
            paired = []
            for i, date in enumerate(sorted_dates):
                target_idx = i + lag
                if target_idx >= len(sorted_dates):
                    continue
                target_date = sorted_dates[target_idx]
                paired.append((date, flare_severity_by_date.get(target_date, 0.0)))

            if len(paired) < self.min_days:
                continue

            flares = np.array([p[1] for p in paired])

            for feat in ENV_FEATURES:
                values = np.array(
                    [records_by_date[p[0]].get(feat, np.nan) for p in paired]
                )
                mask = ~np.isnan(values) & ~np.isnan(flares)
                if mask.sum() < self.min_days:
                    continue

                v, f = values[mask], flares[mask]
                if np.std(v) < 1e-9 or np.std(f) < 1e-9:
                    continue

                threshold = float(np.percentile(v, self.binarize_percentile))
                high = v > threshold
                if high.sum() < 2 or high.sum() > len(v) - 2:
                    continue

                diff_obs = float(f[high].mean() - f[~high].mean())

                # Wrong-direction findings: kept in BH surface with p=1.0
                if diff_obs > 0:
                    p_val = self._perm_p_mean_diff_onesided(high, f, diff_obs)
                else:
                    p_val = 1.0

                candidates.append({
                    "label": f"{feat} (>p{self.binarize_percentile})",
                    "feature_base": feat,
                    "lag": lag,
                    "effect": diff_obs,
                    "p": p_val,
                    "n_obs": int(mask.sum()),
                    "n_exposed": int(high.sum()),
                })

        return candidates

    def _apply_bh_correction(self, candidates: list) -> list:
        """Apply Benjamini-Hochberg FDR correction in place."""
        candidates.sort(key=lambda c: c["p"])
        n = len(candidates)
        q_values = [0.0] * n
        running_min = 1.0
        for k in range(n - 1, -1, -1):
            rank = k + 1
            adjusted = candidates[k]["p"] * n / rank
            running_min = min(running_min, adjusted)
            q_values[k] = min(1.0, running_min)

        patterns = []
        for k, c in enumerate(candidates):
            q = q_values[k]
            if q >= self.fdr_target:
                continue
            patterns.append(DetectedPattern(
                feature=c["label"],
                lag_days=c["lag"],
                effect_size=c["effect"],
                p_value=c["p"],
                q_value=q,
                n_exposed=c["n_exposed"],
                n_total=c["n_obs"],
                confidence=self._confidence(q),
            ))

        patterns.sort(key=lambda p: abs(p.effect_size), reverse=True)
        return patterns

    def _perm_p_mean_diff_onesided(
        self, bool_vec: np.ndarray, flares: np.ndarray, observed_diff: float
    ) -> float:
        """One-sided permutation p-value for mean-difference test."""
        perms = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            shuffled = self.rng.permutation(flares)
            perms[i] = shuffled[bool_vec].mean() - shuffled[~bool_vec].mean()
        return float((perms >= observed_diff).mean())

    @staticmethod
    def _confidence(q: float) -> str:
        """Map q-value to confidence tier."""
        if q < 0.01:
            return "high"
        if q < 0.05:
            return "medium"
        return "low"
