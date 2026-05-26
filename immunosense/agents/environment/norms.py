"""Layer 1 — Regional baseline norms for environmental features.

5 US regions x 4 seasons x 5 features = 100 (mean, std) reference cells.

Sources (approximations - verify before clinical use):
    PM2.5/ozone: EPA AQS 2018-2023 averages
    UV:          WHO Global Solar UV Index regional means
    Barometric:  NOAA climatology, |24h-change| magnitudes
    Pollen:      NAB station data, Universal Pollen Index 0-5
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from scipy.stats import norm

REGIONS = ["NE", "SE", "MW", "W", "SW"]
SEASONS = ["spring", "summer", "fall", "winter"]
FEATURES = ["pm25", "ozone", "uv", "barometric", "pollen"]


REGIONAL_NORMS = {
    "NE": {
        "spring": {"pm25": (8.5, 3.5),  "ozone": (45, 10), "uv": (5.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (5.5, 2.0)},
        "summer": {"pm25": (9.0, 4.0),  "ozone": (55, 12), "uv": (8.0, 1.5),  "barometric": (0.3, 0.2), "pollen": (3.5, 1.5)},
        "fall":   {"pm25": (8.0, 3.5),  "ozone": (40, 9),  "uv": (4.0, 1.5),  "barometric": (0.5, 0.3), "pollen": (4.5, 2.0)},
        "winter": {"pm25": (9.5, 4.0),  "ozone": (32, 7),  "uv": (2.0, 1.0),  "barometric": (0.6, 0.4), "pollen": (1.0, 0.8)},
    },
    "SE": {
        "spring": {"pm25": (9.0, 3.5),  "ozone": (50, 11), "uv": (7.0, 1.5),  "barometric": (0.3, 0.3), "pollen": (7.5, 2.0)},
        "summer": {"pm25": (10.0, 4.5), "ozone": (60, 13), "uv": (9.5, 1.5),  "barometric": (0.3, 0.3), "pollen": (4.0, 1.5)},
        "fall":   {"pm25": (8.5, 3.5),  "ozone": (42, 10), "uv": (5.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (5.5, 2.0)},
        "winter": {"pm25": (9.0, 4.0),  "ozone": (35, 8),  "uv": (3.0, 1.0),  "barometric": (0.5, 0.3), "pollen": (2.0, 1.0)},
    },
    "MW": {
        "spring": {"pm25": (9.5, 4.0),  "ozone": (45, 10), "uv": (5.5, 1.5),  "barometric": (0.5, 0.4), "pollen": (6.0, 2.0)},
        "summer": {"pm25": (10.5, 4.5), "ozone": (58, 12), "uv": (8.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (4.5, 1.5)},
        "fall":   {"pm25": (9.0, 4.0),  "ozone": (40, 9),  "uv": (4.0, 1.5),  "barometric": (0.5, 0.4), "pollen": (5.0, 2.0)},
        "winter": {"pm25": (11.0, 5.0), "ozone": (30, 7),  "uv": (1.5, 0.8),  "barometric": (0.7, 0.5), "pollen": (1.0, 0.8)},
    },
    "W": {
        "spring": {"pm25": (8.0, 4.0),  "ozone": (50, 11), "uv": (6.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (5.0, 2.0)},
        "summer": {"pm25": (12.0, 7.0), "ozone": (62, 14), "uv": (9.0, 1.5),  "barometric": (0.3, 0.2), "pollen": (3.0, 1.5)},
        "fall":   {"pm25": (11.0, 6.5), "ozone": (45, 10), "uv": (4.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (4.0, 2.0)},
        "winter": {"pm25": (9.5, 5.0),  "ozone": (28, 7),  "uv": (2.5, 1.0),  "barometric": (0.6, 0.4), "pollen": (1.5, 1.0)},
    },
    "SW": {
        "spring": {"pm25": (8.5, 3.5),  "ozone": (55, 12), "uv": (7.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (5.5, 2.0)},
        "summer": {"pm25": (10.0, 4.0), "ozone": (68, 14), "uv": (10.5, 1.5), "barometric": (0.3, 0.3), "pollen": (3.5, 1.5)},
        "fall":   {"pm25": (8.5, 3.5),  "ozone": (48, 11), "uv": (5.5, 1.5),  "barometric": (0.4, 0.3), "pollen": (4.5, 2.0)},
        "winter": {"pm25": (8.0, 3.5),  "ozone": (38, 9),  "uv": (3.0, 1.0),  "barometric": (0.5, 0.3), "pollen": (1.5, 1.0)},
    },
}


ZIP_FIRST_DIGIT_TO_REGION = {
    "0": "NE", "1": "NE", "2": "SE", "3": "SE",
    "4": "MW", "5": "MW", "6": "MW",
    "7": "SW", "8": "SW", "9": "W",
}


MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


def infer_region_from_zip(zip_code: Optional[object]) -> str:
    """Infer US region from 5-digit ZIP code's first digit. Defaults to SE if invalid."""
    if zip_code is None:
        return "SE"
    zip_str = str(zip_code).strip().zfill(5)[:5]
    if not zip_str or not zip_str[0].isdigit():
        return "SE"
    return ZIP_FIRST_DIGIT_TO_REGION.get(zip_str[0], "SE")


def infer_season_from_date(date: object) -> str:
    """Infer season from any pandas-parseable date string or object."""
    return MONTH_TO_SEASON[pd.Timestamp(date).month]


def get_population_percentile(region: str, season: str, feature: str, value: float) -> float:
    """Place observed value on regional+seasonal population CDF.

    Args:
        region: One of ``REGIONS``.
        season: One of ``SEASONS``.
        feature: One of ``FEATURES``.
        value: Observed value.

    Returns:
        Percentile in [0, 1] under assumed normal CDF with regional+seasonal
        mean and std.

    Raises:
        ValueError: If region, season, or feature is unknown.
    """
    if region not in REGIONAL_NORMS:
        raise ValueError(f"Unknown region '{region}'")
    if season not in REGIONAL_NORMS[region]:
        raise ValueError(f"Unknown season '{season}'")
    if feature not in REGIONAL_NORMS[region][season]:
        raise ValueError(f"Unknown feature '{feature}'")
    mean, std = REGIONAL_NORMS[region][season][feature]
    if std <= 0:
        return 0.5
    return float(norm.cdf((value - mean) / std))
