"""Meal extraction sources.

Extracts atomic food items from free-form meal descriptions.

Sources:
    MockExtractor          — deterministic regex-based extraction (no API key)
    ClaudeHaikuExtractor   — Claude Haiku via tool-use, with prompt caching
"""

from immunosense.agents.dietary.sources.base import (
    Extractor,
    ExtractedFood,
    ExtractedMeal,
)
from immunosense.agents.dietary.sources.claude_haiku import ClaudeHaikuExtractor
from immunosense.agents.dietary.sources.mock import MockExtractor

__all__ = [
    "Extractor",
    "ExtractedFood",
    "ExtractedMeal",
    "MockExtractor",
    "ClaudeHaikuExtractor",
    "make_default_extractor",
]


def make_default_extractor() -> Extractor:
    """Return a ClaudeHaikuExtractor if API key is set, else MockExtractor."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return ClaudeHaikuExtractor()
        except ImportError:
            pass
    return MockExtractor()
