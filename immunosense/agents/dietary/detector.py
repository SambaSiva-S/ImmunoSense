"""DietaryTriggerDetector — multi-pathway permutation tests for dietary triggers.

For each (feature, lag) combination, tests up to 4 hypotheses:
    - Linear pathway: Pearson r (one-sided positive only)
    - Threshold p75:  Mean-difference binarized at personal p75 (one-sided)
    - Threshold p85:  Mean-difference binarized at personal p85 (one-sided)
    - Threshold p90:  Mean-difference binarized at personal p90 (one-sided)

Reports the strongest path (lowest p-value) that crosses p<0.10. Boolean
features only use mean-difference (no threshold pathways needed).

All p-values via permutation tests (n=500 default). One-sided direction
reflects the biological assumption that triggers CAUSE flares (positive
correlation only).

Note: this detector reports raw permutation p-values. BH FDR correction
is the Conductor's responsibility — Agent 2 deliberately surfaces raw
evidence so the Conductor can do cross-agent corroboration on sub-threshold
patterns (same philosophy as Agent 5's raw_hypothesis_evidence()).
"""

from __future__ import annotations

import numpy as np

from immunosense.agents.dietary.constants import (
    BOOLEAN_TRIGGERS,
    CONTINUOUS_FEATURES,
)
from immunosense.agents.dietary.types import DetectedPattern


class DietaryTriggerDetector:
    """Multi-pathway permutation-test trigger detector.

    Args:
        min_days: Minimum (date, flare) pairs required for a hypothesis to be tested.
        lags: Tuple of lag values in days to test (e.g., (0, 1, 2, 3)).
        n_permutations: Permutation count for empirical p-values.
        alpha: Currently unused (detector reports at p<0.10 threshold).
        binarize_percentiles: Tuple of percentiles to test threshold pathway at.
        random_seed: Seed for reproducible permutation.
    """

    def __init__(
        self,
        min_days: int = 14,
        lags: tuple = (0, 1, 2, 3),
        n_permutations: int = 500,
        alpha: float = 0.05,
        binarize_percentiles: tuple = (75, 85, 90),
        random_seed: int = 42,
    ) -> None:
        self.min_days = min_days
        self.lags = lags
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.binarize_percentiles = binarize_percentiles
        self.rng = np.random.RandomState(random_seed)

    def detect(
        self,
        daily_records: list,
        flare_severity_by_date: dict,
    ) -> list:
        """Run all hypotheses and return patterns crossing p<0.10.

        Args:
            daily_records: List of dicts, each with key 'date' plus per-feature
                            values. Order is preserved by sorting on 'date'.
            flare_severity_by_date: dict mapping date string -> flare severity.
                                     Missing dates are treated as 0.0.

        Returns:
            List of DetectedPattern, sorted by absolute effect size (descending).
            Empty if fewer than min_days records.
        """
        if len(daily_records) < self.min_days:
            return []

        records_by_date = {r["date"]: r for r in daily_records}
        sorted_dates = sorted(records_by_date.keys())
        patterns: list = []

        for lag in self.lags:
            # Build (record_date, flare_on_date+lag) pairs
            paired = []
            for i, date in enumerate(sorted_dates):
                target_idx = i + lag
                if target_idx >= len(sorted_dates):
                    continue
                target_date = sorted_dates[target_idx]
                flare_t = flare_severity_by_date.get(target_date, 0.0)
                paired.append((date, flare_t))

            if len(paired) < self.min_days:
                continue

            flares = np.array([p[1] for p in paired])

            # --- Continuous features (linear + multi-threshold pathways) ---
            for feat in CONTINUOUS_FEATURES:
                values = np.array(
                    [records_by_date[p[0]].get(feat, np.nan) for p in paired]
                )
                mask = ~np.isnan(values) & ~np.isnan(flares)
                if mask.sum() < self.min_days:
                    continue
                v, f = values[mask], flares[mask]
                if np.std(v) < 1e-9 or np.std(f) < 1e-9:
                    continue

                # Collect candidate findings from linear + threshold pathways
                candidates: list = []

                # Linear pathway (Pearson, one-sided positive)
                r_obs = float(np.corrcoef(v, f)[0, 1])
                if r_obs > 0:
                    p_linear = self._perm_p_value_pearson_onesided(v, f, r_obs)
                    candidates.append(("linear", feat, r_obs, p_linear))

                # Threshold pathway — one per percentile (high days vs low days)
                for pct in self.binarize_percentiles:
                    threshold = float(np.percentile(v, pct))
                    high = v > threshold
                    if high.sum() < 2 or high.sum() > len(v) - 2:
                        continue
                    diff_obs = float(f[high].mean() - f[~high].mean())
                    if diff_obs > 0:
                        p_thresh = self._perm_p_value_mean_diff_onesided(
                            high, f, diff_obs,
                        )
                        candidates.append((
                            f"thresh_p{pct}",
                            f"{feat} (>p{pct})",
                            diff_obs,
                            p_thresh,
                        ))

                # Pick the strongest pathway for this (feature, lag)
                if candidates:
                    candidates.sort(key=lambda c: c[3])  # by p-value ascending
                    _best_method, best_label, best_effect, best_p = candidates[0]
                    if best_p < 0.10:
                        patterns.append(DetectedPattern(
                            feature=best_label,
                            lag_days=lag,
                            effect_size=best_effect,
                            p_value=best_p,
                            n_observations=int(mask.sum()),
                            confidence=self._confidence(best_p),
                        ))

            # --- Boolean triggers (one-sided mean-difference) ---
            for feat in BOOLEAN_TRIGGERS:
                values = np.array(
                    [records_by_date[p[0]].get(feat, None) for p in paired]
                )
                mask = np.array([v is not None for v in values]) & ~np.isnan(flares)
                if mask.sum() < self.min_days:
                    continue
                v_bool = np.array(
                    [bool(values[i]) for i in range(len(values)) if mask[i]]
                )
                f = flares[mask]

                if v_bool.sum() < 2 or v_bool.sum() > len(v_bool) - 2:
                    continue

                diff_obs = float(f[v_bool].mean() - f[~v_bool].mean())
                if diff_obs > 0:
                    p_val = self._perm_p_value_mean_diff_onesided(
                        v_bool, f, diff_obs,
                    )
                    if p_val < 0.10:
                        patterns.append(DetectedPattern(
                            feature=feat,
                            lag_days=lag,
                            effect_size=diff_obs,
                            p_value=p_val,
                            n_observations=int(mask.sum()),
                            confidence=self._confidence(p_val),
                        ))

        patterns.sort(key=lambda p: abs(p.effect_size), reverse=True)
        return patterns

    def _perm_p_value_pearson_onesided(
        self,
        values: np.ndarray,
        flares: np.ndarray,
        observed_r: float,
    ) -> float:
        """One-sided permutation p-value for Pearson correlation."""
        permuted_rs = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            shuffled = self.rng.permutation(flares)
            permuted_rs[i] = np.corrcoef(values, shuffled)[0, 1]
        return float((permuted_rs >= observed_r).mean())

    def _perm_p_value_mean_diff_onesided(
        self,
        bool_vec: np.ndarray,
        flares: np.ndarray,
        observed_diff: float,
    ) -> float:
        """One-sided permutation p-value for mean difference (high days - low days)."""
        permuted_diffs = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            shuffled = self.rng.permutation(flares)
            mt = shuffled[bool_vec].mean()
            mf = shuffled[~bool_vec].mean()
            permuted_diffs[i] = mt - mf
        return float((permuted_diffs >= observed_diff).mean())

    @staticmethod
    def _confidence(p: float) -> str:
        """Map p-value to confidence tier."""
        if p < 0.01:
            return "high"
        if p < 0.05:
            return "medium"
        return "low"
