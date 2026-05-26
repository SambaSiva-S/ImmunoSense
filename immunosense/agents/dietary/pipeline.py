"""Layer 2 pipeline: per-meal processing + daily rollup.

`process_meal`: extractor -> matches -> nutrients + triggers + GL (NO DII per-meal)
`rollup_day`:   aggregate meals -> 12-feature Layer 3 vector + daily DII
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from immunosense.agents.dietary.constants import DII_REF
from immunosense.agents.dietary.dii import compute_meal_dii, get_dii_percentile
from immunosense.agents.dietary.matching import FoodMatcher
from immunosense.agents.dietary.sources.base import Extractor
from immunosense.agents.dietary.triggers import classify_food_triggers
from immunosense.agents.dietary.types import DailyRollup, MealResult


def process_meal(
    meal_text: str,
    extractor: Extractor,
    food_matcher: FoodMatcher,
    density_by_code: pd.DataFrame,
    timestamp: Optional[str] = None,
    min_match_score: float = 70.0,
) -> MealResult:
    """Process one meal: text -> nutrients + triggers + glycemic load.

    Per-meal output is NOT a DII score — DII must be computed at the daily
    level (see rollup_day).

    Args:
        meal_text: Free-form meal description.
        extractor: Atomic food extractor (Mock or ClaudeHaiku).
        food_matcher: FoodMatcher with the NHANES food index loaded.
        density_by_code: DataFrame indexed by food_code with nutrient columns.
                          Typically produced by `density.set_index('food_code')`.
        timestamp: Optional ISO8601 timestamp of when the meal was eaten.
        min_match_score: Minimum WRatio match score (0-100) to accept.

    Returns:
        MealResult with nutrients, triggers, glycemic_load, and warnings.
    """
    extracted = extractor.extract(meal_text)
    warnings = list(extracted.extraction_warnings)

    # All nutrient components Layer 3 cares about
    components = list(DII_REF.keys()) + ["sodium_mg"]
    nutrients = {c: 0.0 for c in components}
    triggers = {"dairy": False, "gluten": False, "nightshade": False, "upf": False}
    glycemic_load = 0.0

    if not extracted.foods:
        return MealResult(
            input_text=meal_text,
            timestamp=timestamp,
            extracted_foods=[],
            matches=[],
            nutrients=nutrients,
            triggers=triggers,
            glycemic_load=0.0,
            n_foods_extracted=0,
            n_foods_matched=0,
            n_foods_unmatched=0,
            warnings=warnings + ["No foods extracted"],
        )

    matches = [
        food_matcher.match(f.name, min_score=min_match_score)
        for f in extracted.foods
    ]
    n_matched = sum(1 for m in matches if m is not None)
    n_unmatched = len(matches) - n_matched

    for f, m in zip(extracted.foods, matches):
        if m is None:
            warnings.append(f"No NHANES match for: {f.name}")
        elif m.match_score < 75:
            warnings.append(
                f"Low-confidence match ({m.match_score:.0f}): "
                f"{f.name} -> {m.nhanes_description[:50]}"
            )

    # Accumulate nutrients, triggers, and glycemic load across matched foods
    for food, match in zip(extracted.foods, matches):
        if match is None:
            continue
        try:
            density_row = density_by_code.loc[match.nhanes_code]
        except KeyError:
            warnings.append(
                f"NHANES code {match.nhanes_code} has no nutrient data"
            )
            continue

        scale_factor = food.portion_g / 100.0
        for c in components:
            v = density_row.get(c)
            if pd.notna(v):
                nutrients[c] += float(v) * scale_factor

        cls = classify_food_triggers(match.nhanes_code, match.nhanes_description)
        for trig in triggers:
            if cls[trig]:
                triggers[trig] = True

        # Glycemic load = carb_in_meal * GI / 100
        carb_in_meal = (
            float(density_row.get("carbohydrate_g", 0) or 0) * scale_factor
        )
        glycemic_load += (carb_in_meal * cls["estimated_gi"]) / 100.0

    return MealResult(
        input_text=meal_text,
        timestamp=timestamp,
        extracted_foods=extracted.foods,
        matches=matches,
        nutrients=nutrients,
        triggers=triggers,
        glycemic_load=glycemic_load,
        n_foods_extracted=len(extracted.foods),
        n_foods_matched=n_matched,
        n_foods_unmatched=n_unmatched,
        warnings=warnings,
    )


def compute_intraday_gap_and_timestamps(meal_results: list) -> tuple:
    """Compute first/last meal timestamps and longest within-day gap.

    The within-day gap is NOT the overnight fast — that's computed by
    OvernightFastTracker in Layer 3 using yesterday's last_meal_timestamp.

    Args:
        meal_results: List of MealResult, may include some with None timestamp.

    Returns:
        Tuple (first_ts, last_ts, longest_gap_hours).
        first_ts and last_ts are ISO8601 strings or None.
        longest_gap_hours is float, NaN if fewer than 2 timed meals.
    """
    timed = [m for m in meal_results if m.timestamp is not None]
    if not timed:
        return None, None, float("nan")

    timestamps = []
    for m in timed:
        try:
            timestamps.append(pd.Timestamp(m.timestamp))
        except Exception:
            pass

    if not timestamps:
        return None, None, float("nan")

    timestamps.sort()
    first_ts = timestamps[0].isoformat()
    last_ts = timestamps[-1].isoformat()

    if len(timestamps) < 2:
        return first_ts, last_ts, float("nan")

    gaps = [
        (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0
        for i in range(len(timestamps) - 1)
    ]
    return first_ts, last_ts, float(max(gaps))


def rollup_day(
    meal_results: list,
    date: str,
    age: Optional[float] = None,
    sex: Optional[int] = None,
    bmi: Optional[float] = None,
    dii_models: Optional[dict] = None,
    feature_cols: Optional[list] = None,
) -> DailyRollup:
    """Aggregate a day's meals into the 12-feature Layer 3 vector.

    Args:
        meal_results: List of MealResult for this date.
        date: ISO date string for this rollup.
        age, sex, bmi: Patient demographics for percentile lookup. If any
                       are None or dii_models is None, percentile = None.
        dii_models: Trained Layer 1 quantile regressors (optional).
        feature_cols: Feature column ordering used at Layer 1 training time.

    Returns:
        DailyRollup with daily nutrient totals + 12 Layer-3-facing fields.
        DII is computed on the daily totals (correct mathematical level).
    """
    components = list(DII_REF.keys()) + ["sodium_mg"]
    daily = {c: 0.0 for c in components}
    triggers = {"dairy": False, "gluten": False, "nightshade": False, "upf": False}
    glycemic_load = 0.0

    for mr in meal_results:
        for c in components:
            daily[c] += mr.nutrients.get(c, 0.0)
        for trig in triggers:
            if mr.triggers.get(trig, False):
                triggers[trig] = True
        glycemic_load += mr.glycemic_load

    daily_dii = compute_meal_dii(daily)

    # Omega-6 / Omega-3 ratio (with safety against divide-by-zero)
    o3 = daily.get("omega3_g", 0.0)
    o6 = daily.get("omega6_g", 0.0)
    ratio = (o6 / o3) if o3 > 1e-6 else float("nan")

    first_ts, last_ts, gap = compute_intraday_gap_and_timestamps(meal_results)

    # DII percentile — only computable with demographics + trained models
    percentile = None
    if (
        age is not None
        and sex is not None
        and bmi is not None
        and dii_models is not None
        and not pd.isna(daily_dii)
    ):
        fc = feature_cols if feature_cols is not None else ["age", "sex", "bmi"]
        percentile = get_dii_percentile(
            age, sex, bmi, daily_dii, dii_models=dii_models, feature_cols=fc,
        )

    feature_confidence = {
        "dii_score": "high",
        "omega6_omega3_ratio": "high",
        "sodium_mg": "high",
        "alcohol_g": "high",
        "glycemic_load": "medium",
        "longest_intraday_gap_hours": "medium" if not pd.isna(gap) else "unavailable",
        "first_meal_timestamp": "high" if first_ts else "unavailable",
        "last_meal_timestamp": "high" if last_ts else "unavailable",
        "gluten_present": "medium",
        "dairy_present": "medium",
        "nightshade_present": "medium",
        "upf_present": "low",
    }

    return DailyRollup(
        date=date,
        meal_count=len(meal_results),
        daily_nutrients=daily,
        dii_score=daily_dii,
        omega6_omega3_ratio=ratio,
        glycemic_load=glycemic_load,
        sodium_mg=daily.get("sodium_mg", 0.0),
        alcohol_g=daily.get("alcohol_g", 0.0),
        first_meal_timestamp=first_ts,
        last_meal_timestamp=last_ts,
        longest_intraday_gap_hours=gap,
        gluten_present=triggers["gluten"],
        dairy_present=triggers["dairy"],
        nightshade_present=triggers["nightshade"],
        upf_present=triggers["upf"],
        feature_confidence=feature_confidence,
        daily_dii_percentile=percentile,
        meal_results=meal_results,
    )
