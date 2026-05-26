"""Thermo-Autonomic Decoupling Index (TADI).

TADI measures the breakdown of normal HRV ↔ skin temperature coupling during
sleep. In healthy autonomic regulation, HRV and temperature show coordinated
oscillations. Inflammation/stress break this coupling, producing high TADI.

Computed as lagged Pearson correlation between HRV and temp curves (z-scored
within night), then transformed to: TADI = 1 - max(0, peak_correlation).

Range: 0 (perfect coupling) to 1 (full decoupling).
"""

from __future__ import annotations

import numpy as np


def compute_tadi(
    hrv_curve: np.ndarray,
    temp_curve: np.ndarray,
    min_samples: int = 30,
) -> float:
    """Compute the Thermo-Autonomic Decoupling Index.

    Args:
        hrv_curve: Per-minute HRV values (e.g., rolling RMSSD).
        temp_curve: Per-minute skin temperature values.
        min_samples: Minimum number of paired samples required.

    Returns:
        Float in [0, 1]. 0 = perfect coupling, 1 = full decoupling.
        NaN if insufficient data or zero variance.
    """
    if len(hrv_curve) < min_samples or len(temp_curve) < min_samples:
        return np.nan

    # Align lengths
    n = min(len(hrv_curve), len(temp_curve))
    hrv = hrv_curve[:n]
    temp = temp_curve[:n]

    # Z-score within night
    if np.std(hrv) < 1e-9 or np.std(temp) < 1e-9:
        return np.nan
    hrv_z = (hrv - np.mean(hrv)) / np.std(hrv)
    temp_z = (temp - np.mean(temp)) / np.std(temp)

    # Lagged correlation, lags in [-15, +30] minutes
    best_corr = -1.0
    for lag in range(-15, 31):
        if lag < 0:
            a, b = hrv_z[-lag:], temp_z[:lag]
        elif lag > 0:
            a, b = hrv_z[:-lag], temp_z[lag:]
        else:
            a, b = hrv_z, temp_z
        if len(a) < 10:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isnan(c) and c > best_corr:
            best_corr = c

    return float(1.0 - max(0.0, best_corr))
