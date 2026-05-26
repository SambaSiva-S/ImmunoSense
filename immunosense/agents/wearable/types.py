"""Feature naming constants for Agent 4 (Wearable).

29-feature vector layout (order matters - this defines the vector positions):
    HRV (10):        rmssd_24hr, rmssd_sleep, rmssd_deep_sleep, sdnn, pnn50,
                     hf_power, lf_hf_ratio, sleep_vs_baseline_ratio,
                     trend_6hr, anomaly_score
    Sleep (5):       duration_hrs, deep_pct, rem_pct, efficiency, vs_baseline
    Temp (2):        deviation, trend_3day
    Cardio (3):      resting_hr, vs_baseline, recovery_1min
    Activity (2):    steps_24hr, vs_baseline
    Respiratory (2): spo2_overnight_min, spo2_dips_count
    TADI (2):        index, vs_baseline
    Composite (2):   wearable_stress_score, composite_alert_count
    Quality (1):     data_quality_overall
"""

from __future__ import annotations


FEATURE_NAMES = [
    # HRV (10)
    "hrv_rmssd_24hr",
    "hrv_rmssd_sleep",
    "hrv_rmssd_deep_sleep",
    "hrv_sdnn",
    "hrv_pnn50",
    "hrv_hf_power",
    "hrv_lf_hf_ratio",
    "hrv_sleep_vs_baseline_ratio",
    "hrv_trend_6hr",
    "hrv_anomaly_score",
    # Sleep (5)
    "sleep_duration_hrs",
    "sleep_deep_pct",
    "sleep_rem_pct",
    "sleep_efficiency",
    "sleep_vs_baseline",
    # Temp (2)
    "skin_temp_deviation",
    "skin_temp_trend_3day",
    # Cardio (3)
    "resting_hr",
    "resting_hr_vs_baseline",
    "hr_recovery_1min",
    # Activity (2)
    "activity_steps_24hr",
    "activity_vs_baseline",
    # Respiratory (2)
    "spo2_overnight_min",
    "spo2_dips_count",
    # TADI (2)
    "thermo_autonomic_decoupling_index",
    "tadi_vs_baseline",
    # Composite (2)
    "wearable_stress_score",
    "composite_alert_count",
    # Quality (1)
    "data_quality_overall",
]

# Sanity check
assert len(FEATURE_NAMES) == 29, f"Expected 29 features, got {len(FEATURE_NAMES)}"


# Features the RobustBaselineTracker maintains baselines for (subset of FEATURE_NAMES)
TRACKED_FEATURES = [
    "hrv_rmssd_sleep",
    "hrv_rmssd_deep_sleep",
    "hrv_rmssd_24hr",
    "sleep_efficiency",
    "sleep_duration_hrs",
    "skin_temp_deviation",
    "resting_hr",
    "activity_steps_24hr",
    "thermo_autonomic_decoupling_index",
]
