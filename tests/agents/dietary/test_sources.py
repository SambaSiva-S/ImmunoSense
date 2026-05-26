"""Tests for dietary.sources (MockExtractor mainly; ClaudeHaikuExtractor needs API key)."""

from immunosense.agents.dietary.sources import (
    Extractor,
    ExtractedFood,
    ExtractedMeal,
    MockExtractor,
    make_default_extractor,
)


def test_mock_extractor_implements_protocol():
    """MockExtractor should satisfy the Extractor protocol."""
    m = MockExtractor()
    # Has extract method that takes a string
    assert hasattr(m, "extract")
    out = m.extract("test")
    assert isinstance(out, ExtractedMeal)


def test_mock_eggs_and_toast():
    """Plain breakfast meal."""
    m = MockExtractor()
    result = m.extract("Two scrambled eggs and toast")
    food_names = [f.name for f in result.foods]
    assert "eggs" in food_names
    assert "toast white bread" in food_names


def test_mock_chicken_with_rice():
    m = MockExtractor()
    result = m.extract("Chicken with rice and salad")
    food_names = [f.name for f in result.foods]
    assert "chicken breast cooked" in food_names
    assert "rice white cooked" in food_names
    assert "lettuce salad" in food_names


def test_mock_empty_input():
    """Empty input yields no foods and a warning."""
    m = MockExtractor()
    result = m.extract("")
    assert result.foods == []
    assert len(result.extraction_warnings) > 0


def test_mock_nonsense_input():
    """Nonsense input that doesn't match any rule yields a warning."""
    m = MockExtractor()
    result = m.extract("xyzzy plugh quux")
    assert result.foods == []
    assert "matched no known foods" in result.extraction_warnings[0]


def test_mock_portion_estimates():
    """Mock returns plausible portion sizes."""
    m = MockExtractor()
    result = m.extract("Two scrambled eggs")
    eggs = next((f for f in result.foods if f.name == "eggs"), None)
    assert eggs is not None
    assert 50 < eggs.portion_g < 200
    assert eggs.portion_confidence in ("high", "default", "low")


def test_mock_returns_input_text():
    """The input text should be preserved on the ExtractedMeal."""
    m = MockExtractor()
    text = "Some interesting food I had"
    result = m.extract(text)
    assert result.input_text == text


def test_mock_extracted_food_fields():
    """Each ExtractedFood should have name, portion_g, portion_confidence."""
    m = MockExtractor()
    result = m.extract("apple and banana")
    for food in result.foods:
        assert isinstance(food, ExtractedFood)
        assert isinstance(food.name, str)
        assert isinstance(food.portion_g, (int, float))
        assert food.portion_confidence in ("high", "default", "low")


def test_make_default_extractor_returns_extractor():
    """make_default_extractor returns either Mock or Claude (both implement Extractor)."""
    e = make_default_extractor()
    assert hasattr(e, "extract")
    # Should successfully extract
    out = e.extract("eggs and toast")
    assert isinstance(out, ExtractedMeal)


def test_mock_alcohol_detection():
    """Beer/wine/whiskey/vodka all map to 'beer' (alcohol marker)."""
    m = MockExtractor()
    for drink in ["had a beer", "glass of wine", "shot of vodka"]:
        result = m.extract(drink)
        food_names = [f.name for f in result.foods]
        assert "beer" in food_names, f"Failed on: {drink}"
