"""Dietary builder — raw meal logs -> DailyRollup via the real NHANES pipeline.

Unlike the symptom/biomarker builders (direct mappings), the dietary agent
consumes a DailyRollup, which is the OUTPUT of the dietary Layer 2 pipeline:
    meal text -> extract foods -> match to NHANES -> nutrients -> DII/triggers/GL
    -> rollup to a daily summary.

This builder wires that pipeline. It depends on:
  - NHANES caches (nutrient density + food search index), built ONCE locally
    from P_DR1IFF.XPT and P_DRXFCD.XPT (see build_caches / the README).
  - an Extractor (text -> food list). Default: MockExtractor (deterministic,
    no API). Swap to ClaudeHaiku for higher extraction quality — one arg.

The caches are multi-MB and are NOT bundled; they're generated on the deploy
machine. DietaryPipeline loads them lazily and holds them for reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from immunosense.agents.dietary.density import (
    build_food_search_index,
    build_nutrient_density_cache,
)
from immunosense.agents.dietary.matching import FoodMatcher
from immunosense.agents.dietary.pipeline import process_meal, rollup_day
from immunosense.agents.dietary.sources.mock import MockExtractor
from immunosense.agents.dietary.types import DailyRollup


class DietaryPipeline:
    """Holds the loaded NHANES resources + extractor, runs meals -> DailyRollup.

    Construct once (loads caches), reuse across requests. Thread-safety note:
    the matcher/density are read-only after load, safe to share.
    """

    def __init__(
        self,
        density_cache_path: str | Path,
        food_index_cache_path: str | Path,
        extractor=None,
        min_match_score: float = 70.0,
    ):
        """Args:
        density_cache_path: pickle produced by build_nutrient_density_cache.
        food_index_cache_path: pickle produced by build_food_search_index.
        extractor: text->food extractor. Defaults to MockExtractor. Pass a
            ClaudeHaiku extractor instance for higher-quality extraction.
        min_match_score: minimum fuzzy match score (0-100) to accept a food.
        """
        import pandas as pd

        self._density = pd.read_pickle(Path(density_cache_path))
        self._density_by_code = self._density.set_index("food_code")
        food_index = pd.read_pickle(Path(food_index_cache_path))
        self._matcher = FoodMatcher(food_index)
        self._extractor = extractor or MockExtractor()
        self._min_match_score = min_match_score

    def build_rollup(
        self,
        meal_logs,
        date: str,
        age: Optional[float] = None,
        sex: Optional[int] = None,
        bmi: Optional[float] = None,
    ) -> DailyRollup:
        """Run a day's meal logs through the pipeline into a DailyRollup.

        Args:
            meal_logs: iterable of DietaryLog rows (or objects) exposing
                `description` (the meal text) and optionally `meal_at`.
            date: ISO date string for the rollup.
            age, sex, bmi: demographics for DII percentile (optional).

        Returns:
            DailyRollup ready for the dietary agent.
        """
        meal_results = []
        for log in meal_logs:
            text = getattr(log, "description", None)
            if not text:
                continue
            ts = getattr(log, "meal_at", None)
            ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else (ts or None)
            mr = process_meal(
                meal_text=text,
                extractor=self._extractor,
                food_matcher=self._matcher,
                density_by_code=self._density_by_code,
                timestamp=ts_iso,
                min_match_score=self._min_match_score,
            )
            meal_results.append(mr)

        return rollup_day(
            meal_results=meal_results,
            date=date,
            age=age,
            sex=sex,
            bmi=bmi,
        )


# --------------------------------------------------------------------------- #
# One-time cache building (run locally from the NHANES .XPT files)
# --------------------------------------------------------------------------- #
def build_caches(
    dr1iff_path: str | Path,
    drxfcd_path: str | Path,
    density_cache_path: str | Path,
    food_index_cache_path: str | Path,
    verbose: bool = True,
) -> dict:
    """Build the NHANES caches from the raw .XPT files. Run ONCE per machine.

    Args:
        dr1iff_path: path to NHANES P_DR1IFF.XPT (individual foods).
        drxfcd_path: path to NHANES P_DRXFCD.XPT (food code dictionary).
        density_cache_path / food_index_cache_path: where to write the pickles.

    Returns:
        summary dict with row counts.
    """
    density = build_nutrient_density_cache(
        Path(dr1iff_path), Path(density_cache_path), verbose=verbose
    )
    index = build_food_search_index(
        Path(drxfcd_path), density, Path(food_index_cache_path), verbose=verbose
    )
    return {
        "food_codes": int(density.shape[0]),
        "nutrient_components": int(density.shape[1] - 1),
        "indexed_foods": int(index.shape[0]),
        "density_cache": str(density_cache_path),
        "food_index_cache": str(food_index_cache_path),
    }
