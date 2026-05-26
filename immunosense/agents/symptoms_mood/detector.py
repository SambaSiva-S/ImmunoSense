"""Trigger detector with biologically pre-registered lags.

Pre-registered lag per feature based on biology. 10 hypotheses total.
BH FDR at alpha=0.10 for high-confidence pattern detection.

Additionally exposes raw_hypothesis_evidence() for Conductor cross-agent
corroboration: ALL 10 hypothesis results, including those that don't
survive BH FDR.

Lag convention: feature_on_day(i) paired with flare_severity_on_day(i+lag)
    lag > 0: flare comes AFTER feature (predictive or concurrent)
    lag < 0: flare came BEFORE feature (reactive - feature is consequence)

Lag categories:
    Predictive: sleep (+2), energy (+1) — feature precedes flare
    Concurrent: joint_pain, brain_fog, gi_distress, wellness (+1),
                fatigue, skin (+2) — feature with flare
    Reactive:   phq8, gad7 (-3) — feature appears after flare
"""

from __future__ import annotations

import numpy as np

from immunosense.agents.symptoms_mood.types import (
    DetectedSymptomPattern,
    HypothesisEvidence,
)


# Pre-registered biology-driven lag per feature
PREREGISTERED_FEATURE_LAGS = {
    # Predictive (feature precedes flare)
    "sleep_severity":     2,
    "energy_severity":    1,
    # Concurrent (feature manifests with/right after flare)
    "joint_pain":         1,
    "brain_fog_severity": 1,
    "gi_distress":        1,
    "wellness_severity":  1,
    "fatigue":            2,
    "skin_severity":      2,
    # Reactive (feature appears AFTER flare)
    "phq8_score":        -3,
    "gad7_score":        -3,
}


_PREDICTIVE_FEATS = {"sleep_severity", "energy_severity"}
_REACTIVE_FEATS = {"phq8_score", "gad7_score"}


def _biology_category(feat: str) -> str:
    """Map feature name to biological category."""
    if feat in _PREDICTIVE_FEATS:
        return "predictive"
    if feat in _REACTIVE_FEATS:
        return "reactive"
    return "concurrent"


class SymptomsMoodTriggerDetector:
    """BH FDR-corrected trigger detector with pre-registered biological lags.

    Args:
        min_days: Minimum observation days required.
        n_permutations: Permutations per hypothesis test.
        binarize_percentile: Threshold percentile for high-exposure binarization.
        fdr_target: Target FDR level (default 0.10).
        random_seed: Reproducibility seed.
    """

    def __init__(
        self,
        min_days: int = 14,
        n_permutations: int = 500,
        binarize_percentile: int = 85,
        fdr_target: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        self.min_days = min_days
        self.n_permutations = n_permutations
        self.binarize_percentile = binarize_percentile
        self.fdr_target = fdr_target
        self.rng = np.random.RandomState(random_seed)
        self.last_n_hypotheses: int = 0
        self.last_evidence: list = []

    def detect(
        self,
        daily_records: list,
        flare_severity_by_date: dict,
    ) -> list:
        """Run BH FDR detection. Returns BH-survivors; populates last_evidence with all.

        Args:
            daily_records: List of dicts with 'date' key and feature values.
            flare_severity_by_date: Dict mapping date -> flare severity.

        Returns:
            List of DetectedSymptomPattern (only BH survivors), sorted by |effect| desc.
        """
        if len(daily_records) < self.min_days:
            self.last_n_hypotheses = 0
            self.last_evidence = []
            return []

        records_by_date = {r["date"]: r for r in daily_records}
        sorted_dates = sorted(records_by_date.keys())
        candidates = self._build_candidates(
            sorted_dates, records_by_date, flare_severity_by_date
        )

        self.last_n_hypotheses = len(candidates)
        if not candidates:
            self.last_evidence = []
            return []

        return self._apply_bh_correction(candidates)

    def _build_candidates(
        self,
        sorted_dates: list,
        records_by_date: dict,
        flare_severity_by_date: dict,
    ) -> list:
        """Build candidate hypotheses for each pre-registered (feature, lag)."""
        candidates = []

        for feat, lag in PREREGISTERED_FEATURE_LAGS.items():
            paired = []
            for i, date in enumerate(sorted_dates):
                target_idx = i + lag
                if target_idx < 0 or target_idx >= len(sorted_dates):
                    continue
                target_date = sorted_dates[target_idx]
                paired.append((date, flare_severity_by_date.get(target_date, 0.0)))

            if len(paired) < self.min_days:
                continue

            flares = np.array([p[1] for p in paired], dtype=float)
            raw = [records_by_date[p[0]].get(feat) for p in paired]
            values = np.array([(np.nan if v is None else v) for v in raw], dtype=float)
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
                "feature": feat,
                "label": f"{feat} (>p{self.binarize_percentile})",
                "lag": lag,
                "effect": diff_obs,
                "p": p_val,
                "n_obs": int(mask.sum()),
            })

        return candidates

    def _apply_bh_correction(self, candidates: list) -> list:
        """Apply BH FDR. Populates last_evidence with ALL, returns BH survivors only."""
        candidates.sort(key=lambda c: c["p"])
        n = len(candidates)
        q_values = [0.0] * n
        running_min = 1.0
        for k in range(n - 1, -1, -1):
            rank = k + 1
            adjusted = candidates[k]["p"] * n / rank
            running_min = min(running_min, adjusted)
            q_values[k] = min(1.0, running_min)

        # Build evidence list (ALL hypotheses, including non-survivors)
        self.last_evidence = []
        for k, c in enumerate(candidates):
            q = q_values[k]
            self.last_evidence.append(HypothesisEvidence(
                feature=c["feature"],
                lag_days=c["lag"],
                effect_size=c["effect"],
                raw_p_value=c["p"],
                q_value=q,
                n_observations=c["n_obs"],
                biology_category=_biology_category(c["feature"]),
                survives_fdr=(q < self.fdr_target),
            ))

        # Return only BH-FDR survivors
        patterns = []
        for k, c in enumerate(candidates):
            q = q_values[k]
            if q >= self.fdr_target:
                continue
            patterns.append(DetectedSymptomPattern(
                feature=c["label"],
                lag_days=c["lag"],
                effect_size=c["effect"],
                p_value=c["p"],
                q_value=q,
                n_observations=c["n_obs"],
                confidence=self._confidence(q),
            ))

        patterns.sort(key=lambda p: abs(p.effect_size), reverse=True)
        return patterns

    def _perm_p_mean_diff_onesided(
        self, bool_vec: np.ndarray, flares: np.ndarray, observed_diff: float
    ) -> float:
        """One-sided permutation p-value."""
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
