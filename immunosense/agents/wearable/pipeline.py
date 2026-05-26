"""Layer 3 feature engineering pipeline for Agent 4.

engineer_features produces a 29-feature reading dict from one night of
minute-level data + RR intervals.

apply_baseline_fillin uses the RobustBaselineTracker's personal context to
populate Layer 4 fields (anomaly scores, ratios, trends).

build_output_vector packs the 29 features into a numpy array in canonical order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from immunosense.agents.wearable.hrv import compute_hrv_features
from immunosense.agents.wearable.sleep import compute_sleep_features
from immunosense.agents.wearable.tadi import compute_tadi
from immunosense.agents.wearable.types import FEATURE_NAMES


def engineer_features(
    night_df: pd.DataFrame,
    rr_intervals: list,
    night_idx: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Produce a 29-feature reading dict from one night of data.

    Args:
        night_df: DataFrame with columns: timestamp, hr, skin_temp, enmo, spo2,
            sleep_stage. One row per minute.
        rr_intervals: List/array of RR intervals (ms) for the night.
        night_idx: Index of this night (used as "hour" key).
        rng: Optional Generator for deterministic activity_steps fallback.

    Returns:
        Reading dict with all 29 features (Layer 4 fields set to None).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rr = np.array(rr_intervals)
    n_minutes = len(night_df)
    stage_at_minute = night_df["sleep_stage"].values

    # Sleep-only and deep-sleep-only RR sets (heuristic: split RRs across minutes)
    beats_per_min = max(int(len(rr) / n_minutes), 1)
    rr_sleep, rr_deep = [], []
    for i, st in enumerate(stage_at_minute):
        beats_this_min = rr[i * beats_per_min:(i + 1) * beats_per_min]
        if st in ("light", "deep", "rem"):
            rr_sleep.extend(beats_this_min)
        if st == "deep":
            rr_deep.extend(beats_this_min)

    hrv_all = compute_hrv_features(rr)
    hrv_sleep = compute_hrv_features(np.array(rr_sleep))
    hrv_deep = compute_hrv_features(np.array(rr_deep))

    sleep = compute_sleep_features(stage_at_minute)

    # Resting HR: p10 of sleep-period HR
    sleep_mask = np.isin(stage_at_minute, ["light", "deep", "rem"])
    sleep_hr = night_df.loc[sleep_mask, "hr"].values
    resting_hr = float(np.percentile(sleep_hr, 10)) if len(sleep_hr) > 0 else np.nan

    # Skin temp deviation from a fixed reference (33.5 = healthy adult overnight median)
    overnight_temp = (
        float(np.median(night_df.loc[sleep_mask, "skin_temp"].values))
        if sleep_mask.any() else np.nan
    )
    skin_temp_deviation = (
        overnight_temp - 33.5 if not np.isnan(overnight_temp) else np.nan
    )

    # SpO2 features
    overnight_spo2 = night_df.loc[sleep_mask, "spo2"].values
    spo2_min = (
        float(np.percentile(overnight_spo2, 5)) if len(overnight_spo2) > 0 else np.nan
    )
    spo2_dips = (
        int(np.sum(np.diff(overnight_spo2) < -3)) if len(overnight_spo2) > 1 else 0
    )

    # TADI: HRV curve and temp curve over sleep period (1-min resolution)
    hrv_per_minute = []
    for i in range(n_minutes):
        window_rr = rr[max(0, i - 2) * beats_per_min:(i + 3) * beats_per_min]
        if len(window_rr) >= 10:
            diffs = np.diff(window_rr)
            hrv_per_minute.append(float(np.sqrt(np.mean(diffs ** 2))))
        else:
            hrv_per_minute.append(np.nan)
    hrv_curve = np.array(hrv_per_minute)
    temp_curve = night_df["skin_temp"].values
    # Restrict to sleep minutes
    hrv_curve = hrv_curve[sleep_mask]
    temp_curve = temp_curve[sleep_mask]
    tadi = compute_tadi(hrv_curve, temp_curve)

    # Data quality: fraction of expected features actually computed
    expected = [
        hrv_sleep["rmssd"], hrv_deep["rmssd"], sleep["duration_hrs"],
        skin_temp_deviation, resting_hr, spo2_min, tadi,
    ]
    coverage = float(
        np.mean([1.0 if not np.isnan(v) else 0.0 for v in expected])
    )

    # ---- Build the 29-feature reading dict ----
    reading: dict[str, Any] = {
        "hour": night_idx,
        "timestamp": night_df["timestamp"].iloc[0].isoformat(),
        # HRV (10) — personal-normalized ones left None
        "hrv_rmssd_24hr": hrv_all["rmssd"],
        "hrv_rmssd_sleep": hrv_sleep["rmssd"],
        "hrv_rmssd_deep_sleep": hrv_deep["rmssd"],
        "hrv_sdnn": hrv_sleep["sdnn"],
        "hrv_pnn50": hrv_sleep["pnn50"],
        "hrv_hf_power": hrv_sleep["hf_power"],
        "hrv_lf_hf_ratio": hrv_sleep["lf_hf_ratio"],
        "hrv_sleep_vs_baseline_ratio": None,  # filled by L4
        "hrv_trend_6hr": None,                # filled by L4
        "hrv_anomaly_score": None,            # filled by L4
        # Sleep (5)
        "sleep_duration_hrs": sleep["duration_hrs"],
        "sleep_deep_pct": sleep["deep_pct"],
        "sleep_rem_pct": sleep["rem_pct"],
        "sleep_efficiency": sleep["efficiency"],
        "sleep_vs_baseline": None,            # filled by L4
        # Temp (2)
        "skin_temp_deviation": skin_temp_deviation,
        "skin_temp_trend_3day": None,         # filled by L4
        # Cardio (3)
        "resting_hr": resting_hr,
        "resting_hr_vs_baseline": None,       # filled by L4
        "hr_recovery_1min": None,             # only available if exercise logged
        # Activity (2) — synthetic for now (no daytime data in mock)
        "activity_steps_24hr": float(rng.uniform(6000, 12000)),
        "activity_vs_baseline": None,         # filled by L4
        # Respiratory (2)
        "spo2_overnight_min": spo2_min,
        "spo2_dips_count": spo2_dips,
        # TADI (2)
        "thermo_autonomic_decoupling_index": tadi,
        "tadi_vs_baseline": None,             # filled by L4
        # Composite (2) — filled by L6
        "wearable_stress_score": None,
        "composite_alert_count": 0.0,
        # Quality (1)
        "data_quality_overall": coverage,
    }
    return reading


def apply_baseline_fillin(reading: dict, tracker) -> dict:
    """Use the personal baseline tracker to fill Layer 4 derived fields.

    The tracker should already have been updated with this reading (or older).
    Note: This uses the shared RobustBaselineTracker which exposes per-feature
    data under the key "features" (canonical), not Agent 4's original
    "biomarkers" key.

    Args:
        reading: Feature reading dict (modified in place).
        tracker: A RobustBaselineTracker (or compatible).

    Returns:
        The same reading dict (for fluent chaining).
    """
    ctx = tracker.get_personal_context(reading)

    if not ctx.get("has_personal_data"):
        return reading  # cold start — leave L4 fields as None

    feats = ctx.get("features", {})

    # HRV personal-normalized
    if "hrv_rmssd_sleep" in feats:
        h = feats["hrv_rmssd_sleep"]
        reading["hrv_sleep_vs_baseline_ratio"] = h["ratio"]
        reading["hrv_anomaly_score"] = h["anomaly_score"]

    # Sleep efficiency vs baseline (in IQR units)
    if "sleep_efficiency" in feats:
        reading["sleep_vs_baseline"] = feats["sleep_efficiency"]["anomaly_score"]

    # Resting HR vs baseline
    if "resting_hr" in feats:
        reading["resting_hr_vs_baseline"] = feats["resting_hr"]["anomaly_score"]

    # Activity ratio
    if "activity_steps_24hr" in feats:
        reading["activity_vs_baseline"] = feats["activity_steps_24hr"]["ratio"]

    # TADI vs baseline
    if "thermo_autonomic_decoupling_index" in feats:
        reading["tadi_vs_baseline"] = (
            feats["thermo_autonomic_decoupling_index"]["anomaly_score"]
        )

    return reading


def build_output_vector(reading: dict) -> np.ndarray:
    """Pack 29 features into a numpy array in canonical FEATURE_NAMES order.

    Args:
        reading: Feature reading dict.

    Returns:
        numpy array of shape (29,) with NaN for missing features.
    """
    arr = np.full(29, np.nan, dtype=np.float64)
    for i, name in enumerate(FEATURE_NAMES):
        v = reading.get(name)
        if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
            arr[i] = float(v)
    return arr
