"""Layer 2 preprocessing: artifact removal and signal cleanup.

Three techniques per architecture v6:
    - Incremental Hampel filter — streaming O(log W) artifact removal for RR
    - Akima spline — overshoot-free gap filling for sharp physiological transitions
    - ENMO + variance threshold — sleep/wake refinement from accelerometer
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate


class HampelFilter:
    """Sliding-window Hampel filter for artifact removal.

    Replaces points whose deviation > k*MAD from local median.

    Args:
        window_size: Total window size (will be centered around each point).
        k_threshold: Number of MAD units to use as outlier threshold (default 3.0).
    """

    def __init__(self, window_size: int = 7, k_threshold: float = 3.0) -> None:
        if window_size < 3:
            raise ValueError(f"window_size must be >= 3, got {window_size}")
        if k_threshold <= 0:
            raise ValueError(f"k_threshold must be > 0, got {k_threshold}")
        self.window_size = window_size
        self.k = k_threshold

    def filter(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply Hampel filter to values.

        Args:
            values: 1D array of values.

        Returns:
            Tuple of (filtered_values, outlier_mask).
        """
        values = np.asarray(values, dtype=float)
        n = len(values)
        out = values.copy()
        mask = np.zeros(n, dtype=bool)
        w = self.window_size // 2

        for i in range(n):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            window = values[lo:hi]
            med = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - med))
            if mad > 0 and abs(values[i] - med) > self.k * mad:
                out[i] = med
                mask[i] = True
        return out, mask


def akima_interpolate(
    timestamps_min: np.ndarray,
    values: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Akima spline interpolation: overshoot-free gap filling.

    Use Akima instead of cubic spline to avoid overshoot artifacts that distort
    sharp physiological transitions (e.g., HR drop at sleep onset).

    Args:
        timestamps_min: 1D array of input timestamps (any unit, must match target).
        values: 1D array of values (NaNs treated as missing).
        target_timestamps: 1D array of timestamps to evaluate at.

    Returns:
        1D array of interpolated values at target_timestamps.
    """
    timestamps_min = np.asarray(timestamps_min, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.full(len(target_timestamps), np.nan)
    f = interpolate.Akima1DInterpolator(timestamps_min[valid], values[valid])
    return f(target_timestamps)


def enmo_sleep_wake(
    enmo: np.ndarray,
    hr: np.ndarray,
    enmo_var_window: int = 5,
    enmo_var_threshold: float = 1e-4,
    hr_threshold: float = 75.0,
) -> np.ndarray:
    """Classify each minute as 'sleep' or 'wake' from ENMO + HR.

    Rule: if 5-min rolling ENMO variance is below threshold AND HR is reasonable
    for sleep, label as 'sleep'. Otherwise 'wake'.

    Args:
        enmo: Per-minute ENMO (Euclidean Norm Minus One) accelerometer signal.
        hr: Per-minute heart rate (bpm).
        enmo_var_window: Window size (minutes) for rolling variance.
        enmo_var_threshold: Threshold below which the signal is considered still.
        hr_threshold: HR must be below this for 'sleep' classification.

    Returns:
        Array of string labels: "sleep" or "wake", same length as enmo.
    """
    n = len(enmo)
    labels = np.empty(n, dtype="<U5")
    half = enmo_var_window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        var = float(np.var(enmo[lo:hi]))
        if var < enmo_var_threshold and hr[i] < hr_threshold:
            labels[i] = "sleep"
        else:
            labels[i] = "wake"
    return labels
