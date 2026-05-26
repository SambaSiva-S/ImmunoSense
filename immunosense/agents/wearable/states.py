"""Physiological state derivation from feature readings.

Maps feature values to named semantic state labels for the Conductor and UI:
    - autonomic_state: balanced / severe_vagal_withdrawal / vagal_withdrawal /
                       sympathetic_dominant / mild_sympathetic / balanced
    - sleep_state: restorative / adequate / suboptimal / fragmented / insufficient
    - inflammation_signal: active / elevated / mild_elevation / quiet
    - overall_state: alert / concerning / monitoring / stable
"""

from __future__ import annotations


def derive_physiological_states(reading: dict, alerts: list) -> dict:
    """Map feature values to named semantic state labels.

    Args:
        reading: Feature reading dict.
        alerts: List of alert dicts (with 'severity' key).

    Returns:
        Dict with autonomic_state, sleep_state, inflammation_signal, overall_state.
    """
    # Autonomic state from HRV ratio
    hrv_r = reading.get("hrv_sleep_vs_baseline_ratio")
    if hrv_r is None:
        autonomic = "balanced"
    elif hrv_r < 0.5:
        autonomic = "severe_vagal_withdrawal"
    elif hrv_r < 0.7:
        autonomic = "vagal_withdrawal"
    elif hrv_r < 0.85:
        autonomic = "sympathetic_dominant"
    elif hrv_r < 0.95:
        autonomic = "mild_sympathetic"
    else:
        autonomic = "balanced"

    # Sleep state from efficiency
    se = reading.get("sleep_efficiency", 1.0) or 1.0
    if se >= 0.9:
        sleep = "restorative"
    elif se >= 0.82:
        sleep = "adequate"
    elif se >= 0.7:
        sleep = "suboptimal"
    elif se >= 0.55:
        sleep = "fragmented"
    else:
        sleep = "insufficient"

    # Inflammation signal from temperature deviation
    td = reading.get("skin_temp_deviation", 0.0) or 0.0
    if td > 0.5:
        inflammation = "active"
    elif td > 0.3:
        inflammation = "elevated"
    elif td > 0.15:
        inflammation = "mild_elevation"
    else:
        inflammation = "quiet"

    # Overall state combines alerts + inflammation
    has_critical = any(a.get("severity") == "critical" for a in alerts)
    has_warning = any(a.get("severity") == "warning" for a in alerts)
    if has_critical:
        overall = "alert"
    elif has_warning:
        overall = "concerning"
    elif inflammation in ("active", "elevated"):
        overall = "monitoring"
    else:
        overall = "stable"

    return {
        "autonomic_state": autonomic,
        "sleep_state": sleep,
        "inflammation_signal": inflammation,
        "overall_state": overall,
    }
