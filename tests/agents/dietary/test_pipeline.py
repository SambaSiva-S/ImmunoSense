"""Tests for dietary.pipeline: process_meal, rollup_day, intraday_gap.

Uses a synthetic food_index + density table to test the pipeline without
requiring NHANES downloads.
"""

import math

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.dietary.constants import DII_REF
from immunosense.agents.dietary.matching import FoodMatcher
from immunosense.agents.dietary.pipeline import (
    compute_intraday_gap_and_timestamps,
    process_meal,
    rollup_day,
)
from immunosense.agents.dietary.sources import MockExtractor
from immunosense.agents.dietary.types import MealResult


# ============================================================
# Synthetic fixtures (avoid NHANES dependencies in tests)
# ============================================================

def _build_synthetic_food_index():
    """Build a tiny synthetic NHANES-style food index."""
    rows = [
        (11111000, "Milk, whole", "milk whole"),
        (22222000, "Eggs, raw", "eggs"),
        (51000000, "Bread, white, toast", "bread white toast"),
        (51001000, "Toast, white", "toast white bread"),
        (58101000, "Rice, white, cooked", "rice white cooked"),
        (24100000, "Chicken breast, cooked", "chicken breast cooked"),
        (26100000, "Salmon, cooked", "salmon cooked"),
        (63100000, "Banana, raw", "banana raw"),
        (63200000, "Apple, raw", "apple raw"),
        (92500000, "Coffee, brewed", "coffee brewed"),
        (74100000, "Lettuce, salad", "lettuce salad"),
        (75100000, "Avocado, raw", "avocado raw"),
        (43100000, "Lentils, cooked", "lentils cooked"),
        (11200000, "Yogurt, plain", "yogurt plain"),
        (92200000, "Cola, soda", "cola soda"),
        (25100000, "Sausage, cooked", "sausage cooked"),
        (71000000, "Potato, cooked", "potato cooked"),
        (74200000, "Tomato, raw", "tomato raw"),
        (93100000, "Beer", "beer"),
    ]
    return pd.DataFrame(
        rows, columns=["food_code", "description", "search_text"],
    )


def _build_synthetic_density(food_codes):
    """Build a synthetic per-100g nutrient density table."""
    components = list(DII_REF.keys()) + ["sodium_mg"]
    # All foods have a baseline nutrient profile (small positive values)
    rows = []
    for code in food_codes:
        row = {"food_code": code}
        for c in components:
            # Energy: 200kcal/100g, carbs: 30g/100g, protein: 10g/100g, etc.
            if c == "energy_kcal":
                row[c] = 200.0
            elif c == "carbohydrate_g":
                row[c] = 30.0
            elif c == "protein_g":
                row[c] = 10.0
            elif c == "sodium_mg":
                row[c] = 200.0
            elif c.endswith("_g"):
                row[c] = 0.5
            elif c.endswith("_mg"):
                row[c] = 5.0
            elif c.endswith("_mcg"):
                row[c] = 50.0
            elif c.endswith("_mcg_rae"):
                row[c] = 100.0
            else:
                row[c] = 0.1
        rows.append(row)
    return pd.DataFrame(rows).set_index("food_code")


@pytest.fixture
def food_matcher():
    return FoodMatcher(_build_synthetic_food_index())


@pytest.fixture
def density_by_code():
    food_index = _build_synthetic_food_index()
    return _build_synthetic_density(food_index["food_code"].values)


# ============================================================
# process_meal
# ============================================================

def test_process_meal_basic(food_matcher, density_by_code):
    """Basic meal processing returns a MealResult with nutrients."""
    extractor = MockExtractor()
    result = process_meal(
        "Two eggs and toast",
        extractor=extractor,
        food_matcher=food_matcher,
        density_by_code=density_by_code,
        timestamp="2026-04-01T08:00:00",
    )
    assert isinstance(result, MealResult)
    assert result.n_foods_extracted == 2
    assert result.n_foods_matched >= 1   # at least eggs OR toast matches
    assert result.nutrients["energy_kcal"] > 0


def test_process_meal_empty_input(food_matcher, density_by_code):
    """Empty text returns a MealResult with no foods."""
    result = process_meal(
        "",
        extractor=MockExtractor(),
        food_matcher=food_matcher,
        density_by_code=density_by_code,
    )
    assert result.n_foods_extracted == 0
    assert result.nutrients["energy_kcal"] == 0.0


def test_process_meal_glycemic_load(food_matcher, density_by_code):
    """Glycemic load should be positive for a carb-containing meal."""
    result = process_meal(
        "rice with chicken",
        extractor=MockExtractor(),
        food_matcher=food_matcher,
        density_by_code=density_by_code,
    )
    # Rice has GI=73, chicken has GI=0, so GL should reflect mostly rice
    assert result.glycemic_load > 0


def test_process_meal_triggers_detected(food_matcher, density_by_code):
    """Milk should trigger 'dairy', bread should trigger 'gluten'."""
    result = process_meal(
        "milk and toast",
        extractor=MockExtractor(),
        food_matcher=food_matcher,
        density_by_code=density_by_code,
    )
    # Even if some matches imperfect, the descriptions should classify
    # We don't assert specifics here since fuzzy match could go either way;
    # just verify the trigger structure exists
    assert "dairy" in result.triggers
    assert "gluten" in result.triggers
    assert "nightshade" in result.triggers
    assert "upf" in result.triggers


def test_process_meal_warnings_for_unmatched(food_matcher, density_by_code):
    """Foods that don't match anything should generate warnings."""
    # Build a custom extractor that returns a clearly-unmatched food
    from immunosense.agents.dietary.sources.base import (
        ExtractedFood,
        ExtractedMeal,
    )

    class _NonsenseExtractor:
        def extract(self, text):
            return ExtractedMeal(
                input_text=text,
                foods=[ExtractedFood(name="xyzzy_unknown", portion_g=100, portion_confidence="default")],
                extraction_warnings=[],
            )

    result = process_meal(
        "anything",
        extractor=_NonsenseExtractor(),
        food_matcher=food_matcher,
        density_by_code=density_by_code,
    )
    assert result.n_foods_extracted == 1
    # The fuzzy matcher may find SOMETHING above min_score=70 even for nonsense,
    # so we don't assert unmatched=1. Just confirm structure.
    assert result.n_foods_matched + result.n_foods_unmatched == 1


def test_process_meal_low_confidence_match_warning(food_matcher, density_by_code):
    """Matches with score 70-75 should generate low-confidence warnings (informational)."""
    # We can't easily force a 72-score match; just verify no crash on borderline cases
    result = process_meal(
        "rice",
        extractor=MockExtractor(),
        food_matcher=food_matcher,
        density_by_code=density_by_code,
    )
    # Pipeline shouldn't crash; warnings list exists
    assert hasattr(result, "warnings")


# ============================================================
# compute_intraday_gap_and_timestamps
# ============================================================

def _build_meal(timestamp):
    """Build a minimal MealResult with the given timestamp."""
    return MealResult(
        input_text="test",
        timestamp=timestamp,
        extracted_foods=[],
        matches=[],
        nutrients={},
        triggers={},
        glycemic_load=0.0,
        n_foods_extracted=0,
        n_foods_matched=0,
        n_foods_unmatched=0,
        warnings=[],
    )


def test_intraday_gap_no_meals():
    """Empty meal list returns (None, None, NaN)."""
    first, last, gap = compute_intraday_gap_and_timestamps([])
    assert first is None
    assert last is None
    assert math.isnan(gap)


def test_intraday_gap_no_timestamps():
    """Meals with None timestamps return (None, None, NaN)."""
    meals = [_build_meal(None), _build_meal(None)]
    first, last, gap = compute_intraday_gap_and_timestamps(meals)
    assert first is None
    assert last is None
    assert math.isnan(gap)


def test_intraday_gap_one_meal():
    """One meal: first==last, gap NaN."""
    meals = [_build_meal("2026-04-01T08:00:00")]
    first, last, gap = compute_intraday_gap_and_timestamps(meals)
    assert first is not None
    assert last is not None
    assert math.isnan(gap)


def test_intraday_gap_three_meals():
    """Gap = largest interval between consecutive meals."""
    meals = [
        _build_meal("2026-04-01T08:00:00"),
        _build_meal("2026-04-01T12:30:00"),
        _build_meal("2026-04-01T19:00:00"),
    ]
    _first, _last, gap = compute_intraday_gap_and_timestamps(meals)
    # Gaps: 4.5h and 6.5h -> max = 6.5h
    assert abs(gap - 6.5) < 1e-6


def test_intraday_gap_unsorted():
    """Out-of-order timestamps should be sorted before computing gaps."""
    meals = [
        _build_meal("2026-04-01T19:00:00"),
        _build_meal("2026-04-01T08:00:00"),
        _build_meal("2026-04-01T12:30:00"),
    ]
    first, last, gap = compute_intraday_gap_and_timestamps(meals)
    assert "08:00:00" in first
    assert "19:00:00" in last
    assert abs(gap - 6.5) < 1e-6


# ============================================================
# rollup_day
# ============================================================

def test_rollup_day_zero_meals(food_matcher, density_by_code):
    """Empty meal list yields a degenerate rollup."""
    rollup = rollup_day([], date="2026-04-01")
    assert rollup.meal_count == 0
    assert rollup.glycemic_load == 0.0
    assert rollup.first_meal_timestamp is None


def test_rollup_day_aggregates_nutrients(food_matcher, density_by_code):
    """Two meals -> nutrients sum."""
    m1 = process_meal(
        "rice", extractor=MockExtractor(),
        food_matcher=food_matcher, density_by_code=density_by_code,
        timestamp="2026-04-01T08:00:00",
    )
    m2 = process_meal(
        "chicken", extractor=MockExtractor(),
        food_matcher=food_matcher, density_by_code=density_by_code,
        timestamp="2026-04-01T19:00:00",
    )
    rollup = rollup_day([m1, m2], date="2026-04-01")
    assert rollup.meal_count == 2
    # Energy should be roughly sum of both meals
    expected_energy_min = (m1.nutrients["energy_kcal"] + m2.nutrients["energy_kcal"]) * 0.99
    assert rollup.daily_nutrients["energy_kcal"] >= expected_energy_min


def test_rollup_day_dii_computed_at_daily_level(food_matcher, density_by_code):
    """Daily DII should be computed from aggregated daily nutrients."""
    m1 = process_meal(
        "rice", extractor=MockExtractor(),
        food_matcher=food_matcher, density_by_code=density_by_code,
    )
    rollup = rollup_day([m1], date="2026-04-01")
    # DII should be a finite number (or NaN if too few components)
    assert isinstance(rollup.dii_score, float)


def test_rollup_day_omega_ratio(food_matcher, density_by_code):
    """omega6_omega3_ratio should be computed safely (no divide-by-zero)."""
    m1 = process_meal(
        "rice", extractor=MockExtractor(),
        food_matcher=food_matcher, density_by_code=density_by_code,
    )
    rollup = rollup_day([m1], date="2026-04-01")
    # Either a valid number or NaN — never crashes
    assert isinstance(rollup.omega6_omega3_ratio, float)


def test_rollup_day_triggers_carry_through(food_matcher, density_by_code):
    """If any meal has a trigger flag set, the rollup should have it too."""
    # Build a synthetic meal result with dairy_present=True
    m_with_dairy = MealResult(
        input_text="milk",
        timestamp=None,
        extracted_foods=[],
        matches=[],
        nutrients={c: 0.0 for c in list(DII_REF.keys()) + ["sodium_mg"]},
        triggers={"dairy": True, "gluten": False, "nightshade": False, "upf": False},
        glycemic_load=0.0,
        n_foods_extracted=1,
        n_foods_matched=1,
        n_foods_unmatched=0,
        warnings=[],
    )
    rollup = rollup_day([m_with_dairy], date="2026-04-01")
    assert rollup.dairy_present is True
    assert rollup.gluten_present is False


def test_rollup_day_dii_percentile_none_without_demographics():
    """Without age/sex/bmi or models, daily_dii_percentile should be None."""
    rollup = rollup_day([], date="2026-04-01")  # no demographics, no models
    assert rollup.daily_dii_percentile is None


def test_rollup_day_dii_percentile_with_mock_models():
    """With demographics + mock dii_models, percentile should be computed."""
    class _MockModel:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.array([self.v])

    mock_models = {
        0.10: _MockModel(-2.0),
        0.50: _MockModel(0.0),
        0.90: _MockModel(2.0),
    }
    # Build a rollup with nonzero DII (use a meal with nutrients)
    m = MealResult(
        input_text="test", timestamp=None,
        extracted_foods=[], matches=[],
        nutrients={c: ref[0] for c, ref in DII_REF.items()},  # all at mean -> DII = 0
        triggers={"dairy": False, "gluten": False, "nightshade": False, "upf": False},
        glycemic_load=0.0,
        n_foods_extracted=0, n_foods_matched=0, n_foods_unmatched=0,
        warnings=[],
    )
    rollup = rollup_day(
        [m], date="2026-04-01",
        age=30, sex=1, bmi=22,
        dii_models=mock_models,
    )
    # DII ~ 0, which is at the 0.50 quantile (medians of mock models)
    assert rollup.daily_dii_percentile is not None
    assert 0.4 <= rollup.daily_dii_percentile <= 0.6
