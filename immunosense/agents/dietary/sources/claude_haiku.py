"""ClaudeHaikuExtractor - production meal extraction via Claude API.

Uses Claude Haiku with:
  - Tool use (structured JSON output via `extract_foods` tool)
  - Ephemeral prompt caching on the system prompt (saves cost on repeated calls)

Decomposes composite dishes into atomic ingredients matching NHANES granularity.
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.common.anthropic_client import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_MAX_TOKENS,
    get_anthropic_client,
)
from immunosense.agents.dietary.sources.base import (
    ExtractedFood,
    ExtractedMeal,
)


_SYSTEM_PROMPT = """You are a dietary intake extraction system for a clinical \
research pipeline. Your job: decompose free-form meal descriptions into atomic \
food components matching the granularity of the USDA/NHANES food database.

Rules:
1. Decompose composite dishes into atomic ingredients. \
"chicken biryani" -> ["cooked basmati rice", "cooked chicken breast", "ghee", \
"fried onions", "mixed spices"]. Do NOT emit "biryani" as a single item.
2. Use generic, search-friendly names. "white bread toast" not "Wonder Bread \
slice". Match what would appear in a dietary database.
3. Estimate portion in grams. If the patient specified a portion ("two slices", \
"one cup"), convert: 1 slice bread ~28g, 1 cup cooked rice ~180g, \
1 cup milk ~240g, 1 medium apple ~180g, 1 medium banana ~120g, \
1 cup cooked pasta ~200g, 1 can soda ~355g, 1 beer ~355g, 1 glass wine ~150g.
4. portion_confidence: "high" when patient gave explicit quantity, "default" \
when you used a standard portion, "low" when ambiguous ("a big bowl", "some chicken").
5. Skip water, condiments under 5g (salt, pepper, herbs as garnish), beverages \
that are pure water.
6. If meal description is empty, nonsensical, or not food, return an empty foods list.

Output via the extract_foods tool. Never respond in free text."""


_TOOL_SCHEMA = {
    "name": "extract_foods",
    "description": "Emit the decomposed atomic food list for the meal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "foods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "portion_g": {"type": "number"},
                        "portion_confidence": {
                            "type": "string",
                            "enum": ["high", "default", "low"],
                        },
                    },
                    "required": ["name", "portion_g", "portion_confidence"],
                },
            },
        },
        "required": ["foods"],
    },
}


class ClaudeHaikuExtractor:
    """Production extractor: Claude Haiku with prompt caching + tool use.

    Args:
        api_key: Override API key. If None, reads ANTHROPIC_API_KEY env var.
        model: Override model ID. Defaults to the shared DEFAULT_EXTRACTION_MODEL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is None:
            self._client = get_anthropic_client(api_key=self.api_key)
        return self._client

    def extract(self, meal_text: str) -> ExtractedMeal:
        """Extract atomic food list from meal text. Never raises; returns warnings."""
        if not meal_text or not meal_text.strip():
            return ExtractedMeal(
                input_text=meal_text, foods=[],
                extraction_warnings=["empty meal text"],
            )

        try:
            client = self._get_client()
        except (RuntimeError, ImportError) as e:
            return ExtractedMeal(
                input_text=meal_text, foods=[],
                extraction_warnings=[f"Client setup failed: {e}"],
            )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=[_TOOL_SCHEMA],
                tool_choice={"type": "tool", "name": "extract_foods"},
                messages=[{"role": "user", "content": meal_text}],
            )
        except Exception as e:
            return ExtractedMeal(
                input_text=meal_text, foods=[],
                extraction_warnings=[
                    f"API call failed: {type(e).__name__}: {e}"
                ],
            )

        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            return ExtractedMeal(
                input_text=meal_text, foods=[],
                extraction_warnings=["LLM did not emit a tool_use block"],
            )

        foods = [
            ExtractedFood(
                name=f["name"],
                portion_g=float(f["portion_g"]),
                portion_confidence=f["portion_confidence"],
            )
            for f in tool_block.input.get("foods", [])
        ]
        return ExtractedMeal(
            input_text=meal_text, foods=foods, extraction_warnings=[],
        )
