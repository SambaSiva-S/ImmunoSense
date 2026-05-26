"""Base types for meal extraction sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ExtractedFood:
    """One atomic food item extracted from a meal description.

    Attributes:
        name: Generic search-friendly food name (e.g., 'cooked white rice')
        portion_g: Estimated portion size in grams
        portion_confidence: 'high' if patient gave explicit quantity,
                            'default' if standard portion assumed,
                            'low' if ambiguous ('a big bowl')
    """

    name: str
    portion_g: float
    portion_confidence: str   # 'high' | 'default' | 'low'


@dataclass
class ExtractedMeal:
    """All foods extracted from one meal description."""

    input_text: str
    foods: list  # list[ExtractedFood]
    extraction_warnings: list = field(default_factory=list)


class Extractor(Protocol):
    """Protocol every meal extractor must implement."""

    def extract(self, meal_text: str) -> ExtractedMeal:
        ...
