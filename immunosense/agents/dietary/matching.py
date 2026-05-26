"""Food name -> NHANES code matching.

Two-stage fuzzy match:
    Stage 1: Prefix-anchored within food whose first word stem matches the query
    Stage 2: Global fuzzy fallback over entire NHANES food corpus

Uses rapidfuzz.WRatio scorer (handles short/abbreviated queries well).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.dietary.types import FoodMatch


def normalize_text(s: str) -> str:
    """Strip diacritics + lowercase + punctuation -> spaces + collapse whitespace.

    Used both to build the search corpus and to canonicalize incoming queries.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_food_search_index(cache_path: Path) -> pd.DataFrame:
    """Load a previously-built food search index from disk.

    The index has three columns: food_code (int64), description (str),
    search_text (str, normalize_text(description)).

    Args:
        cache_path: Path to the pickled DataFrame.

    Returns:
        DataFrame with columns: food_code, description, search_text

    Raises:
        FileNotFoundError if the cache doesn't exist (caller should
        build it via build_food_search_index in density module).
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Food index not found at {cache_path}. "
            "Build it first via dietary.density.build_food_search_index() "
            "or run `python -m immunosense.agents.dietary.layer1_train`."
        )
    return pd.read_pickle(cache_path)


class FoodMatcher:
    """Holds the food search index and exposes match_food().

    Decoupled from module-level state so multiple matchers can coexist
    (e.g., separate test fixtures).
    """

    def __init__(self, food_index: pd.DataFrame) -> None:
        """
        Args:
            food_index: DataFrame from load_food_search_index().
                         Must have columns: food_code, description, search_text.
        """
        required_cols = {"food_code", "description", "search_text"}
        missing = required_cols - set(food_index.columns)
        if missing:
            raise ValueError(f"food_index missing required columns: {missing}")

        self.food_index = food_index.reset_index(drop=True)
        # Cache derived arrays for performance
        self._corpus = self.food_index["search_text"].tolist()
        self._code_array = self.food_index["food_code"].values
        self._desc_array = self.food_index["description"].values

    def match(self, extracted_name: str, min_score: float = 70.0) -> Optional[FoodMatch]:
        """Fuzzy-match an extracted food name to NHANES food code.

        Args:
            extracted_name: Free-form food name from the extractor.
            min_score: Minimum WRatio match score (0-100) to accept.

        Returns:
            FoodMatch with the best match, or None if nothing crosses min_score.
        """
        from rapidfuzz import fuzz, process

        query = normalize_text(extracted_name)
        if not query:
            return None

        first_word = query.split()[0]

        # Stage 1: prefix-anchored — restrict to foods whose first word stem
        # matches the query's first word stem
        stems = {first_word}
        if first_word.endswith("s") and len(first_word) > 3:
            stems.add(first_word[:-1])

        prefix_mask = self.food_index["search_text"].str.split().str[0].isin(stems)
        candidates = self.food_index[prefix_mask]

        if len(candidates) > 0:
            corpus = candidates["search_text"].tolist()
            codes = candidates["food_code"].values
            descs = candidates["description"].values

            result = process.extractOne(query, corpus, scorer=fuzz.WRatio)
            if result is not None:
                _text, score, idx = result
                if score >= min_score:
                    return FoodMatch(
                        extracted_name=extracted_name,
                        nhanes_code=int(codes[idx]),
                        nhanes_description=str(descs[idx]),
                        match_score=float(score),
                    )

        # Stage 2: global fallback over the entire corpus
        result = process.extractOne(query, self._corpus, scorer=fuzz.WRatio)
        if result is None:
            return None
        _text, score, idx = result
        if score < min_score:
            return None

        return FoodMatch(
            extracted_name=extracted_name,
            nhanes_code=int(self._code_array[idx]),
            nhanes_description=str(self._desc_array[idx]),
            match_score=float(score),
        )


def match_food(
    extracted_name: str,
    food_index: pd.DataFrame,
    min_score: float = 70.0,
) -> Optional[FoodMatch]:
    """Stateless convenience wrapper around FoodMatcher.match().

    Builds a FoodMatcher each call — for repeated calls, instantiate
    FoodMatcher once and call .match() instead (faster).
    """
    matcher = FoodMatcher(food_index)
    return matcher.match(extracted_name, min_score=min_score)
