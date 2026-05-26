"""Layer 2 pipeline and Conductor-facing flare signature.

process_environment_day(zip_code, target_date, source) -> DailyEnvironmentSummary
    Layer 2 orchestration: fetch + percentile + threshold classification.

compute_flare_signature(daily_summary, anomaly_scores, detected_patterns) -> dict
    Conductor-facing 0-1 environmental flare risk score with contributing factors.

Evidence weights (from design doc literature review):
    PM2.5      = 0.35 (strongest evidence base)
    UV         = 0.20 (SLE/dermatomyositis photosensitivity)
    Ozone      = 0.15
    Barometric = 0.15 (RA joint pain, Smedslund 2010)
    Pollen     = 0.15

Per-patient detected triggers boost the effective weight 1.5x.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from immunosense.agents.environment.norms import (
    get_population_percentile,
    infer_region_from_zip,
    infer_season_from_date,
)
from immunosense.agents.environment.sources.composite import CompositeEnvironmentSource
from immunosense.agents.environment.thresholds import classify_all_thresholds
from immunosense.agents.environment.types import (
    DailyEnvironmentSummary,
    EnvironmentDataSource,
)


# Approximate ZIP -> (lat, lon) for commonly-tested locations
ZIP_APPROX_COORDS = {
    "28202": (35.227, -80.843),    # Charlotte, NC
    "85001": (33.448, -112.074),   # Phoenix, AZ
    "02101": (42.358, -71.060),    # Boston, MA
    "60601": (41.886, -87.625),    # Chicago, IL
    "90001": (33.974, -118.249),   # Los Angeles, CA
    "10001": (40.751, -74.000),    # New York, NY
    "94102": (37.779, -122.419),   # San Francisco, CA
    "77001": (29.749, -95.358),    # Houston, TX
    "33101": (25.789, -80.224),    # Miami, FL
    "80201": (39.741, -104.984),   # Denver, CO
}


REGION_CENTERS = {
    "NE": (42.0, -75.0),
    "SE": (35.0, -82.0),
    "MW": (40.0, -88.0),
    "W":  (38.0, -120.0),
    "SW": (34.0, -106.0),
}


# Map FetchedFeatures field names to Layer 1 feature names
_FF_TO_L1 = {
    "pm25_ug_m3": "pm25",
    "ozone_ppb": "ozone",
    "uv_index": "uv",
    "barometric_change_kpa": "barometric",
    "pollen_index": "pollen",
}


# Evidence weights for the conductor-facing flare signature
EVIDENCE_WEIGHTS = {
    "pm25_ug_m3":            0.35,
    "uv_index":              0.20,
    "ozone_ppb":             0.15,
    "barometric_change_kpa": 0.15,
    "pollen_index":          0.15,
}


def zip_to_latlon(zip_code: object) -> tuple:
    """Convert ZIP code to (lat, lon). Uses known coords first, then region center."""
    zip_str = str(zip_code).zfill(5)[:5]
    if zip_str in ZIP_APPROX_COORDS:
        return ZIP_APPROX_COORDS[zip_str]
    region = infer_region_from_zip(zip_str)
    return REGION_CENTERS.get(region, (35.0, -82.0))


def process_environment_day(
    zip_code: str,
    target_date: str,
    source: Optional[EnvironmentDataSource] = None,
) -> DailyEnvironmentSummary:
    """Layer 2 pipeline. ZIP + date -> structured summary for Layer 3.

    Args:
        zip_code: 5-digit US ZIP code (string or int).
        target_date: Date string in 'YYYY-MM-DD' format.
        source: Override data source. If None, uses CompositeEnvironmentSource()
            which provides real APIs with mock fallback.

    Returns:
        DailyEnvironmentSummary with raw features, percentiles, threshold alerts,
        and overall confidence.
    """
    if source is None:
        source = CompositeEnvironmentSource()

    lat, lon = zip_to_latlon(zip_code)
    region = infer_region_from_zip(zip_code)
    season = infer_season_from_date(target_date)

    fetched = source.fetch(lat, lon, target_date)

    percentiles = {}
    for ff_field, l1_name in _FF_TO_L1.items():
        value = getattr(fetched, ff_field)
        percentiles[l1_name] = (
            None
            if value is None
            else get_population_percentile(region, season, l1_name, value)
        )

    threshold_alerts = classify_all_thresholds(fetched)

    n_real = sum(1 for v in fetched.confidence.values() if v == "real")
    overall_confidence = n_real / 5.0 if fetched.confidence else 0.0

    return DailyEnvironmentSummary(
        date=target_date,
        location={
            "zip": str(zip_code).zfill(5)[:5],
            "lat": lat,
            "lon": lon,
            "region": region,
            "season": season,
        },
        pm25_ug_m3=fetched.pm25_ug_m3,
        ozone_ppb=fetched.ozone_ppb,
        uv_index=fetched.uv_index,
        barometric_change_kpa=fetched.barometric_change_kpa,
        pollen_index=fetched.pollen_index,
        percentiles=percentiles,
        threshold_alerts=threshold_alerts,
        feature_confidence=dict(fetched.confidence),
        sources=dict(fetched.sources),
        errors=list(fetched.errors),
        overall_confidence=overall_confidence,
    )


def compute_flare_signature(
    daily_summary: DailyEnvironmentSummary,
    anomaly_scores: dict,
    detected_patterns: Optional[list] = None,
) -> dict:
    """Conductor-facing 0-1 environmental flare risk score.

    Args:
        daily_summary: Today's DailyEnvironmentSummary.
        anomaly_scores: Per-feature anomaly scores from the agent's tracker.
        detected_patterns: Optional list of DetectedPattern for personal-trigger boost.

    Returns:
        Dict with:
            score: 0-1 environmental flare risk
            contributing_factors: list of per-feature contribution dicts
            threshold_breaches: list of feature names hitting unhealthy thresholds
            data_quality_confidence: overall_confidence from daily_summary
    """
    detected_patterns = detected_patterns or []

    triggered_features = set()
    for p in detected_patterns:
        base = p.feature.split(" ")[0]  # 'pm25_ug_m3 (>p85)' -> 'pm25_ug_m3'
        if p.confidence in ("high", "medium"):
            triggered_features.add(base)

    contributions = []
    total_score = 0.0

    for feature, weight in EVIDENCE_WEIGHTS.items():
        anomaly = anomaly_scores.get(feature, float("nan"))
        if pd.isna(anomaly):
            normalized = 0.0
        else:
            normalized = max(0.0, min(3.0, anomaly)) / 3.0  # clip + normalize to [0,1]

        effective_weight = weight * (1.5 if feature in triggered_features else 1.0)
        contribution = normalized * effective_weight
        total_score += contribution

        contributions.append({
            "feature": feature,
            "anomaly_score": float(anomaly) if not pd.isna(anomaly) else None,
            "normalized": normalized,
            "evidence_weight": weight,
            "effective_weight": effective_weight,
            "is_personal_trigger": feature in triggered_features,
        })

    total_score = min(1.0, total_score)
    contributions.sort(
        key=lambda c: c["normalized"] * c["effective_weight"], reverse=True
    )

    # Threshold categories considered "breaches" for the flare signature
    BREACH_CATEGORIES = {
        "unhealthy_sensitive", "unhealthy", "very_unhealthy", "hazardous",
        "high", "very_high", "extreme", "large_change",
    }

    return {
        "score": total_score,
        "contributing_factors": contributions,
        "threshold_breaches": [
            f for f, alert in daily_summary.threshold_alerts.items()
            if alert in BREACH_CATEGORIES
        ],
        "data_quality_confidence": daily_summary.overall_confidence,
    }
