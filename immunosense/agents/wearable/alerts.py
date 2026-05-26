"""Layer 5 alert engine: single-metric and composite alerts.

Rule schemas:
    Single-metric: feature + threshold + comparison + severity + rule_type
    Composite:    name + min_conditions_met + time_window + list of conditions

Rules live as Python dicts here. In production these may move to YAML
(agent4_rules.yaml) but the data shape is unchanged.

Note: This uses the shared RobustBaselineTracker which exposes per-feature
data under the key "features" (not Agent 4's original "biomarkers" key).
"""

from __future__ import annotations

import uuid

import numpy as np


SINGLE_METRIC_RULES = [
    {
        "name": "hrv_critical_low",
        "feature": "hrv_rmssd_sleep",
        "threshold": 20.0,
        "comparison": "less_than",
        "severity": "critical",
        "rule_type": "population",
    },
    {
        "name": "hrv_personal_drop",
        "feature": "hrv_rmssd_sleep",
        "threshold_ratio": 0.7,  # of baseline
        "comparison": "less_than",
        "severity": "warning",
        "rule_type": "personal",
        "min_baseline_days": 14,
    },
    {
        "name": "spo2_overnight_low",
        "feature": "spo2_overnight_min",
        "threshold": 92.0,
        "comparison": "less_than",
        "severity": "warning",
        "rule_type": "population",
    },
    {
        "name": "skin_temp_elevated",
        "feature": "skin_temp_deviation",
        "threshold": 2.0,  # anomaly_score units, not raw C
        "comparison": "greater_than",
        "severity": "warning",
        "rule_type": "personal",  # personal anomaly score, not raw value
    },
    {
        "name": "resting_hr_elevated",
        "feature": "resting_hr_vs_baseline",
        "threshold": 2.0,
        "comparison": "greater_than",
        "severity": "warning",
        "rule_type": "personal",
    },
]


COMPOSITE_RULES = [
    {
        "name": "autoimmune_prodrome",
        "severity": "critical",
        "min_conditions_met": 3,
        "time_window_nights": 2,
        "conditions": [
            {"feature": "hrv_rmssd_sleep", "personal_ratio_lt": 0.75},
            {"feature": "sleep_efficiency", "personal_anomaly_lt": -1.5},
            {"feature": "skin_temp_deviation", "value_gt": 0.3},
            {"feature": "thermo_autonomic_decoupling_index", "personal_anomaly_gt": 2.0},
        ],
    },
    {
        "name": "acute_stress_response",
        "severity": "warning",
        "min_conditions_met": 2,
        "time_window_nights": 1,
        "conditions": [
            {"feature": "hrv_rmssd_sleep", "personal_ratio_lt": 0.7},
            {"feature": "resting_hr_vs_baseline", "value_gt": 2.0},
        ],
    },
]


def evaluate_single_metric_alerts(
    reading: dict,
    tracker,
    rules_version: str = "2026.05.24-001",
) -> list:
    """Fire single-metric alerts based on the reading + baseline state.

    Args:
        reading: Feature reading dict.
        tracker: A RobustBaselineTracker (or compatible) instance.
        rules_version: Stamp included in each alert for audit.

    Returns:
        List of alert dicts.
    """
    alerts = []
    ctx = tracker.get_personal_context(reading)
    feats = ctx.get("features", {})

    for rule in SINGLE_METRIC_RULES:
        feat = rule["feature"]
        value = reading.get(feat)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        triggered = False
        if rule["rule_type"] == "population":
            threshold = rule["threshold"]
            triggered = (
                (rule["comparison"] == "less_than" and value < threshold)
                or (rule["comparison"] == "greater_than" and value > threshold)
            )
        else:  # personal
            if feat not in feats:
                continue
            if ctx.get("readings_count", 0) < rule.get("min_baseline_days", 14):
                continue
            baseline = feats[feat]["median_baseline"]
            if "threshold_ratio" in rule:
                threshold = rule["threshold_ratio"] * baseline
                triggered = (
                    (rule["comparison"] == "less_than" and value < threshold)
                    or (rule["comparison"] == "greater_than" and value > threshold)
                )
            else:
                # threshold compared against personal anomaly_score (sigma units)
                anomaly = feats[feat]["anomaly_score"]
                triggered = (
                    (rule["comparison"] == "less_than" and anomaly < rule["threshold"])
                    or (rule["comparison"] == "greater_than" and anomaly > rule["threshold"])
                )

        if triggered:
            alerts.append({
                "alert_id": f"a-{uuid.uuid4().hex[:8]}",
                "name": rule["name"],
                "severity": rule["severity"],
                "alert_type": "single_metric",
                "feature": feat,
                "value": float(value),
                "rule_type": rule["rule_type"],
                "rules_version": rules_version,
            })
    return alerts


def evaluate_composite_alerts(
    reading: dict,
    tracker,
    trajectory_window: list,
    rules_version: str = "2026.05.24-001",
) -> list:
    """Fire composite (named pattern) alerts.

    Evaluates conditions against the current reading, leveraging the tracker's
    personal context (anomaly_score, ratio) and the trajectory window for
    multi-night patterns.

    Args:
        reading: Current feature reading dict.
        tracker: RobustBaselineTracker (or compatible) instance.
        trajectory_window: Recent readings (for future multi-night logic).
        rules_version: Stamp included in each alert for audit.

    Returns:
        List of composite alert dicts.
    """
    alerts = []
    ctx = tracker.get_personal_context(reading)
    feats = ctx.get("features", {})

    for rule in COMPOSITE_RULES:
        conds_met = 0
        contributing = []
        for cond in rule["conditions"]:
            feat = cond["feature"]
            value = reading.get(feat)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue

            if "value_gt" in cond and value > cond["value_gt"]:
                conds_met += 1
                contributing.append(feat)
            elif "value_lt" in cond and value < cond["value_lt"]:
                conds_met += 1
                contributing.append(feat)
            elif "personal_ratio_lt" in cond and feat in feats:
                if feats[feat]["ratio"] < cond["personal_ratio_lt"]:
                    conds_met += 1
                    contributing.append(feat)
            elif "personal_anomaly_lt" in cond and feat in feats:
                if feats[feat]["anomaly_score"] < cond["personal_anomaly_lt"]:
                    conds_met += 1
                    contributing.append(feat)
            elif "personal_anomaly_gt" in cond and feat in feats:
                if feats[feat]["anomaly_score"] > cond["personal_anomaly_gt"]:
                    conds_met += 1
                    contributing.append(feat)

        if conds_met >= rule["min_conditions_met"]:
            alerts.append({
                "alert_id": f"a-{uuid.uuid4().hex[:8]}",
                "name": rule["name"],
                "severity": rule["severity"],
                "alert_type": "composite",
                "conditions_met": conds_met,
                "total_conditions": len(rule["conditions"]),
                "contributing_features": contributing,
                "rules_version": rules_version,
            })

    return alerts
