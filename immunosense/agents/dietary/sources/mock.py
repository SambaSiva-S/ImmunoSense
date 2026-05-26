"""MockExtractor - deterministic offline meal extraction.

Uses simple regex rules to match common foods. Used when no API key is
available and for testing.
"""

from __future__ import annotations

import re

from immunosense.agents.dietary.sources.base import (
    ExtractedFood,
    ExtractedMeal,
)


class MockExtractor:
    """Deterministic offline extractor for testing without API key."""

    # (pattern, canonical_name, default_portion_g, confidence_level)
    _RULES = [
        (r"\b(scrambled\s+)?eggs?\b",                "eggs",                  100, "default"),
        (r"\btoast\b",                                "toast white bread",     50,  "default"),
        (r"\bpasta\b",                                "pasta cooked",          200, "default"),
        (r"\brice\b",                                 "rice white cooked",     180, "default"),
        (r"\bchicken(\s+(breast|biryani|tikka))?\b",  "chicken breast cooked", 100, "default"),
        (r"\bsalmon\b",                               "salmon cooked",         120, "default"),
        (r"\bbanana\b",                               "banana raw",            120, "default"),
        (r"\bapple\b",                                "apple raw",             150, "default"),
        (r"\b(coffee|cup\s+of\s+coffee)\b",           "coffee brewed",         240, "default"),
        (r"\bmilk\b",                                 "milk whole",            240, "default"),
        (r"\bsalad\b",                                "lettuce salad",         100, "default"),
        (r"\bavocado\b",                              "avocado raw",           80,  "default"),
        (r"\b(dal|lentils?)\b",                       "lentils cooked",        150, "default"),
        (r"\b(yogurt|raita)\b",                       "yogurt plain",          150, "default"),
        (r"\b(soda|cola)\b",                          "cola soda",             355, "default"),
        (r"\b(hot\s+dog|sausage|bacon)\b",            "sausage cooked",        50,  "default"),
        (r"\b(potato|french\s+fries|fries)\b",        "potato cooked",         150, "default"),
        (r"\btomato\b",                               "tomato raw",            100, "default"),
        (r"\b(beer|wine|whiskey|vodka)\b",            "beer",                  355, "default"),
    ]

    def extract(self, meal_text: str) -> ExtractedMeal:
        """Match foods via regex rules. Returns ExtractedMeal with warnings if empty."""
        text = meal_text.lower()
        foods = []
        for pattern, canonical, portion_g, conf in self._RULES:
            if re.search(pattern, text):
                foods.append(ExtractedFood(
                    name=canonical, portion_g=portion_g, portion_confidence=conf,
                ))
        warnings = [] if foods else ["MockExtractor matched no known foods"]
        return ExtractedMeal(
            input_text=meal_text, foods=foods, extraction_warnings=warnings,
        )
