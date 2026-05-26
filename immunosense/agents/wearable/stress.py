"""Composite wearable stress score.

Logistic blend of normalized deficits. Range [0, 1].

Component weights:
    HRV deficit:         35%
    Sleep deficit:       25%
    Temp elevation:      20%
    HR elevation:        15%
    Activity drop:        5%

Weights renormalize when components are missing (NaN inputs are dropped).
"""

from __future__ import annotations

import numpy as np


def compute_wearable_stress_score(reading: dict) -> float:
    """Compute the composite wearable stress score from a feature reading dict.

    Args:
        reading: Feature reading dict (may have None for missing values).

    Returns:
        Score in [0.0, 1.0]. Returns 0.0 if no components are available.
    """
    components = []
    weights = []

    # HRV deficit (35%)
    r = reading.get("hrv_sleep_vs_baseline_ratio")
    if r is not None:
        components.append(max(0.0, min(1.0, 1.0 - r)))
        weights.append(0.35)

    # Sleep deficit (25%)
    se = reading.get("sleep_efficiency")
    if se is not None:
        components.append(max(0.0, min(1.0, 1.0 - se / 0.85)))
        weights.append(0.25)

    # Temp elevation (20%)
    td = reading.get("skin_temp_deviation")
    if td is not None:
        components.append(max(0.0, min(1.0, td / 0.5)))
        weights.append(0.20)

    # HR elevation (15%)
    hr = reading.get("resting_hr_vs_baseline")
    if hr is not None:
        components.append(max(0.0, min(1.0, hr / 3.0)))
        weights.append(0.15)

    # Activity drop (5%)
    a = reading.get("activity_vs_baseline")
    if a is not None:
        components.append(max(0.0, min(1.0, 1.0 - a)))
        weights.append(0.05)

    if not components:
        return 0.0
    weights_arr = np.array(weights) / np.sum(weights)  # renormalize if some missing
    score = float(np.dot(components, weights_arr))
    return max(0.0, min(1.0, score))
