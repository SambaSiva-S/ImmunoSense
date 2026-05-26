"""HRV (Heart Rate Variability) feature computation.

Time-domain + frequency-domain + non-linear HRV features from RR intervals.

Uses Lomb-Scargle periodogram for frequency-domain features rather than
Welch's PSD. Lomb-Scargle handles irregular RR sampling correctly without
requiring resampling/interpolation, which is the correct approach for
beat-to-beat data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle


def compute_hrv_features(rr_intervals: np.ndarray) -> dict:
    """Compute HRV time-domain + frequency-domain + non-linear features.

    Args:
        rr_intervals: Array of RR intervals in milliseconds.

    Returns:
        Dict with keys: rmssd, sdnn, pnn50, hf_power, lf_hf_ratio, sd1, sd2.
        Values are NaN if insufficient data.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    rr = rr[np.isfinite(rr) & (rr > 300) & (rr < 2000)]
    if len(rr) < 30:
        return {
            "rmssd": np.nan,
            "sdnn": np.nan,
            "pnn50": np.nan,
            "hf_power": np.nan,
            "lf_hf_ratio": np.nan,
            "sd1": np.nan,
            "sd2": np.nan,
        }

    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(rr))
    pnn50 = float(100.0 * np.mean(np.abs(diffs) > 50))

    # Frequency domain via Lomb-Scargle (handles irregular RR sampling)
    # Cumulative time of each beat in seconds
    t = np.cumsum(rr) / 1000.0
    freqs = np.linspace(0.04, 0.4, 200)
    angular = 2 * np.pi * freqs
    try:
        psd = lombscargle(t, rr - rr.mean(), angular, normalize=True)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask]))
        hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask]))
        lf_hf_ratio = lf_power / hf_power if hf_power > 1e-9 else np.nan
    except Exception:
        hf_power, lf_hf_ratio = np.nan, np.nan

    # Poincare plot SD1, SD2
    sd1 = float(np.std(diffs) / np.sqrt(2))
    sd2 = (
        float(np.sqrt(2 * sdnn ** 2 - sd1 ** 2))
        if sdnn ** 2 > sd1 ** 2 / 2 else np.nan
    )

    return {
        "rmssd": rmssd,
        "sdnn": sdnn,
        "pnn50": pnn50,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "sd1": sd1,
        "sd2": sd2,
    }
