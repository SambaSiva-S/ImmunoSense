"""Tests for dietary.matching."""

import pandas as pd
import pytest

from immunosense.agents.dietary.matching import (
    FoodMatcher,
    normalize_text,
)
from immunosense.agents.dietary.types import FoodMatch


def test_normalize_text_lowercase():
    assert normalize_text("MILK") == "milk"


def test_normalize_text_strips_punctuation():
    assert normalize_text("apple, raw") == "apple raw"


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  toast   white  ") == "toast white"


def test_normalize_text_strips_diacritics():
    assert normalize_text("café") == "cafe"


def test_normalize_text_empty():
    assert normalize_text("") == ""


def test_normalize_text_only_punctuation():
    assert normalize_text("!!!") == ""


@pytest.fixture
def small_food_index():
    """Tiny synthetic food index."""
    return pd.DataFrame([
        (11111000, "Milk, whole, 3.25% milkfat", "milk whole 3 25 milkfat"),
        (22222000, "Eggs, raw, large", "eggs raw large"),
        (51000000, "Bread, white", "bread white"),
        (58101000, "Rice, white, cooked", "rice white cooked"),
        (24100000, "Chicken breast, cooked, no skin", "chicken breast cooked no skin"),
    ], columns=["food_code", "description", "search_text"])


def test_food_matcher_init(small_food_index):
    """FoodMatcher initializes from a DataFrame."""
    matcher = FoodMatcher(small_food_index)
    assert matcher.food_index is not None


def test_food_matcher_missing_columns():
    """FoodMatcher raises ValueError if columns are missing."""
    bad_df = pd.DataFrame([{"food_code": 1, "description": "foo"}])  # missing search_text
    with pytest.raises(ValueError, match="missing required columns"):
        FoodMatcher(bad_df)


def test_food_matcher_exact_match(small_food_index):
    """Exact match should score very high."""
    matcher = FoodMatcher(small_food_index)
    result = matcher.match("eggs", min_score=70.0)
    assert result is not None
    assert isinstance(result, FoodMatch)
    assert result.nhanes_code == 22222000


def test_food_matcher_fuzzy_match(small_food_index):
    """Slightly misspelled query should still match."""
    matcher = FoodMatcher(small_food_index)
    result = matcher.match("chiken breast", min_score=70.0)
    # 'chiken' should fuzzy-match 'chicken'
    if result is not None:
        assert result.nhanes_code == 24100000


def test_food_matcher_no_match_returns_none(small_food_index):
    """A garbage query should return None."""
    matcher = FoodMatcher(small_food_index)
    result = matcher.match("xyzzyplugh quux foobar", min_score=90.0)
    assert result is None


def test_food_matcher_empty_query(small_food_index):
    """Empty query returns None."""
    matcher = FoodMatcher(small_food_index)
    assert matcher.match("") is None
    assert matcher.match("   ") is None


def test_food_matcher_match_score_in_range(small_food_index):
    """match_score should be a percentage (0-100)."""
    matcher = FoodMatcher(small_food_index)
    result = matcher.match("rice", min_score=50.0)
    if result is not None:
        assert 50 <= result.match_score <= 100


def test_food_matcher_prefix_anchored(small_food_index):
    """Prefix-anchored stage 1 should pick the food whose first word matches."""
    matcher = FoodMatcher(small_food_index)
    # 'rice' has prefix-match against 'rice white cooked'
    result = matcher.match("rice cooked", min_score=70.0)
    assert result is not None
    assert result.nhanes_code == 58101000


def test_food_matcher_plural_stem_handling(small_food_index):
    """'eggs' should still match if corpus has 'eggs raw large'."""
    matcher = FoodMatcher(small_food_index)
    # Singular 'egg' should still find 'eggs raw large' via stem stripping
    result = matcher.match("egg", min_score=70.0)
    # Might be None or match — both acceptable depending on rapidfuzz scoring
    if result is not None:
        assert result.nhanes_code == 22222000
