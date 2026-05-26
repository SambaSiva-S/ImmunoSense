"""Per-food trigger classification.

Given a NHANES food code and its English description, classify whether the
food contains common autoimmune trigger categories: dairy, gluten, nightshades,
ultra-processed food (UPF). Also estimates glycemic index.

Logic combines two signals:
  1. NHANES food code first digit (rough USDA category)
  2. Description keyword matching (the more reliable signal)
"""

from __future__ import annotations

from immunosense.agents.dietary.constants import (
    DAIRY_FIRST_DIGITS,
    DAIRY_KEYWORDS,
    DEFAULT_GI,
    GI_BY_CATEGORY,
    GLUTEN_FIRST_DIGITS,
    GLUTEN_FREE_KEYWORDS,
    GLUTEN_KEYWORDS,
    NIGHTSHADE_EXCLUSIONS,
    NIGHTSHADE_KEYWORDS,
    UPF_KEYWORDS,
)
from immunosense.agents.dietary.matching import normalize_text


def _desc_contains(desc_norm: str, keyword_set: set) -> bool:
    """True if any keyword appears as substring of normalized description."""
    return any(kw in desc_norm for kw in keyword_set)


def classify_food_triggers(nhanes_code: int, description: str) -> dict:
    """Classify one food into trigger categories and estimate glycemic index.

    Args:
        nhanes_code: NHANES food code (8-digit integer)
        description: NHANES English description for the food

    Returns:
        dict with keys: dairy, gluten, nightshade, upf (all bool),
                        estimated_gi (int, glycemic index)
    """
    desc_norm = normalize_text(description)
    first_digit = int(str(nhanes_code)[0]) if nhanes_code > 0 else 0

    # Dairy: USDA category 1 OR matches a dairy keyword
    dairy = (
        first_digit in DAIRY_FIRST_DIGITS
        or _desc_contains(desc_norm, DAIRY_KEYWORDS)
    )

    # Gluten: USDA category 5 OR matches a gluten keyword, BUT NOT if it
    # contains a gluten-free signal (rice, corn, "gluten free" tag)
    has_gluten_kw = _desc_contains(desc_norm, GLUTEN_KEYWORDS)
    has_gf_kw = _desc_contains(desc_norm, GLUTEN_FREE_KEYWORDS)
    gluten = (first_digit in GLUTEN_FIRST_DIGITS or has_gluten_kw) and not has_gf_kw
    # Explicit override: even if the food is in category 5, "gluten free" labeling wins
    if "gluten free" in desc_norm or "gluten-free" in desc_norm:
        gluten = False

    # Nightshade: matches a nightshade keyword AND not an exclusion (sweet potato)
    nightshade = (
        _desc_contains(desc_norm, NIGHTSHADE_KEYWORDS)
        and not _desc_contains(desc_norm, NIGHTSHADE_EXCLUSIONS)
    )

    # UPF: matches any UPF keyword
    upf = _desc_contains(desc_norm, UPF_KEYWORDS)

    # Glycemic index: pick the LONGEST matching keyword
    # (e.g. 'white rice' should beat 'rice' if both present)
    gi = DEFAULT_GI
    matched_kw = ""
    for kw, gi_val in GI_BY_CATEGORY.items():
        if kw in desc_norm and len(kw) > len(matched_kw):
            gi = gi_val
            matched_kw = kw

    return {
        "dairy": dairy,
        "gluten": gluten,
        "nightshade": nightshade,
        "upf": upf,
        "estimated_gi": gi,
    }
