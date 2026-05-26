"""Sleep architecture feature computation from per-minute stage labels."""

from __future__ import annotations

import numpy as np


def compute_sleep_features(stages: np.ndarray) -> dict:
    """Compute sleep features from per-minute sleep stage labels.

    Args:
        stages: Array of stage labels ("light", "deep", "rem", "wake").

    Returns:
        Dict with duration_hrs, deep_pct, rem_pct, efficiency.
    """
    n = len(stages)
    if n == 0:
        return {
            "duration_hrs": np.nan,
            "deep_pct": np.nan,
            "rem_pct": np.nan,
            "efficiency": np.nan,
        }
    asleep = np.isin(stages, ["light", "deep", "rem"]).sum()
    deep = (stages == "deep").sum()
    rem = (stages == "rem").sum()
    return {
        "duration_hrs": asleep / 60.0,
        "deep_pct": deep / max(asleep, 1),
        "rem_pct": rem / max(asleep, 1),
        "efficiency": asleep / n,
    }
