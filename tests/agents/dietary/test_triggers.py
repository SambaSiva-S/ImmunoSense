"""Tests for dietary.triggers.classify_food_triggers."""

import pytest

from immunosense.agents.dietary.triggers import classify_food_triggers


# === Dairy ===

def test_classify_dairy_first_digit():
    """NHANES code starting with 1 (dairy category)."""
    result = classify_food_triggers(11111000, "milk, whole, 3.25% milkfat")
    assert result["dairy"] is True


def test_classify_dairy_keyword_yogurt():
    result = classify_food_triggers(99999000, "yogurt, plain, low fat")
    assert result["dairy"] is True


def test_classify_dairy_keyword_cheese():
    result = classify_food_triggers(99999000, "Cheddar cheese, sliced")
    assert result["dairy"] is True


def test_classify_no_dairy():
    result = classify_food_triggers(58101000, "rice, white, cooked")
    assert result["dairy"] is False


# === Gluten ===

def test_classify_gluten_first_digit():
    """NHANES code starting with 5 (grain category)."""
    result = classify_food_triggers(51000000, "wheat bread")
    assert result["gluten"] is True


def test_classify_gluten_keyword_pasta():
    result = classify_food_triggers(99999000, "Pasta, spaghetti, cooked")
    assert result["gluten"] is True


def test_classify_gluten_free_rice():
    """Rice has GLUTEN_FREE_KEYWORD; cancels grain-category flag."""
    result = classify_food_triggers(58101000, "white rice cooked")
    assert result["gluten"] is False


def test_classify_gluten_free_explicit():
    """Explicit 'gluten free' label always wins."""
    result = classify_food_triggers(51000000, "gluten free bread")
    assert result["gluten"] is False


def test_classify_corn_is_gluten_free():
    """Corn is a gluten-free grain."""
    result = classify_food_triggers(56204000, "corn tortilla")
    # corn is a GLUTEN_FREE_KEYWORD, but tortilla is a GLUTEN_KEYWORD; gluten-free wins
    assert result["gluten"] is False


# === Nightshade ===

def test_classify_nightshade_tomato():
    result = classify_food_triggers(99999000, "raw tomato slice")
    assert result["nightshade"] is True


def test_classify_nightshade_potato():
    result = classify_food_triggers(99999000, "baked potato with skin")
    assert result["nightshade"] is True


def test_classify_sweet_potato_excluded():
    """Sweet potato is botanically distinct, must NOT be flagged as nightshade."""
    result = classify_food_triggers(99999000, "sweet potato, baked")
    assert result["nightshade"] is False


def test_classify_no_nightshade():
    result = classify_food_triggers(99999000, "broccoli, raw")
    assert result["nightshade"] is False


# === UPF ===

def test_classify_upf_soda():
    result = classify_food_triggers(99999000, "cola soda, 12 fl oz")
    assert result["upf"] is True


def test_classify_upf_hot_dog():
    result = classify_food_triggers(99999000, "hot dog with bun")
    assert result["upf"] is True


def test_classify_upf_chocolate_bar():
    result = classify_food_triggers(99999000, "milk chocolate bar")
    assert result["upf"] is True


def test_classify_no_upf_whole_food():
    result = classify_food_triggers(99999000, "apple, raw, with skin")
    assert result["upf"] is False


# === GI ===

def test_gi_white_rice_high():
    result = classify_food_triggers(99999000, "white rice cooked")
    assert result["estimated_gi"] >= 70


def test_gi_apple_low():
    result = classify_food_triggers(99999000, "apple, raw, with skin")
    assert result["estimated_gi"] < 40


def test_gi_chicken_zero():
    result = classify_food_triggers(99999000, "Chicken breast, baked")
    assert result["estimated_gi"] == 0


def test_gi_unknown_food_default():
    """An unknown food should get DEFAULT_GI."""
    result = classify_food_triggers(99999000, "exotic_unknown_food_blah")
    assert result["estimated_gi"] == 50


def test_gi_longest_match_wins():
    """'white rice' should beat 'rice' when both match."""
    result = classify_food_triggers(99999000, "white rice cooked")
    assert result["estimated_gi"] == 73  # 'white rice' specifically


# === Return shape ===

def test_returns_required_keys():
    """Result must include all expected keys."""
    result = classify_food_triggers(99999000, "test food")
    for k in ["dairy", "gluten", "nightshade", "upf", "estimated_gi"]:
        assert k in result


def test_zero_food_code():
    """Edge: food_code = 0 (no first digit)."""
    result = classify_food_triggers(0, "milk")
    # Should still classify by keyword
    assert result["dairy"] is True
