"""Build / load NHANES nutrient density cache and food search index.

Two caches are produced from NHANES Layer 1 data files:

1. **Nutrient density cache** (`nutrient_density_per_100g_v2.pkl`)
   For each NHANES food code, the average per-100g intake values across all
   participants who consumed it. Used to compute per-meal nutrient totals
   given a portion size.

2. **Food search index** (`food_code_search_index.pkl`)
   Lookup table: food_code -> normalized search text. Used by matching.py.

These are built once (offline, takes ~30s) and persisted. Runtime code
loads them, never rebuilds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from immunosense.agents.dietary.constants import NHANES_DR1IFF_COL_MAP
from immunosense.agents.dietary.matching import normalize_text


def build_nutrient_density_cache(
    dr1iff_path: Path,
    cache_path: Path,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build (or load cached) per-100g nutrient table from NHANES P_DR1IFF.

    Args:
        dr1iff_path: Path to NHANES P_DR1IFF.XPT (individual foods file)
        cache_path: Where to read/write the pickled cache
        force_rebuild: If True, ignore existing cache and rebuild
        verbose: Print progress messages

    Returns:
        DataFrame with columns: food_code (int64), and one column per
        nutrient component (float64). Per-100g values, aggregated across
        all NHANES participants who consumed each food code.
    """
    if cache_path.exists() and not force_rebuild:
        if verbose:
            print(f"Cache exists: {cache_path}")
        return pd.read_pickle(cache_path)

    if verbose:
        print("Building nutrient density cache (one-time, ~30s)...")

    df_iff = pd.read_sas(dr1iff_path, format="xport")

    # Filter 1: only reliable recalls (DR1DRSTZ == 1)
    if "DR1DRSTZ" in df_iff.columns:
        df_iff = df_iff[df_iff["DR1DRSTZ"] == 1].copy()

    # Filter 2: must have nonzero gram weight
    df_iff = df_iff[df_iff["DR1IGRMS"] > 0].copy()

    # Compute per-100g scaling factor: 100 / portion_g
    nutrients_per_100g = pd.DataFrame({"food_code": df_iff["DR1IFDCD"].values})
    scale = (100.0 / df_iff["DR1IGRMS"]).values

    # Fill each component column
    for component, iff_col in NHANES_DR1IFF_COL_MAP.items():
        if iff_col in df_iff.columns:
            nutrients_per_100g[component] = df_iff[iff_col].values * scale
        else:
            nutrients_per_100g[component] = np.nan

    # Omega-3 and Omega-6 are computed from individual fatty acid columns
    omega3_cols = [
        c for c in ["DR1IP183", "DR1IP205", "DR1IP225", "DR1IP226"]
        if c in df_iff.columns
    ]
    omega6_cols = [
        c for c in ["DR1IP182", "DR1IP204"] if c in df_iff.columns
    ]
    if omega3_cols:
        nutrients_per_100g["omega3_g"] = (
            df_iff[omega3_cols].sum(axis=1).values * scale
        )
    if omega6_cols:
        nutrients_per_100g["omega6_g"] = (
            df_iff[omega6_cols].sum(axis=1).values * scale
        )

    # NHANES reports caffeine in mg; convert to g for DII consistency
    nutrients_per_100g["caffeine_g"] = nutrients_per_100g["caffeine_g"] / 1000.0

    # Aggregate to one row per food code (mean across all consumption records)
    density = nutrients_per_100g.groupby("food_code").mean().reset_index()
    density["food_code"] = density["food_code"].astype("int64")
    for col in density.columns:
        if col != "food_code":
            density[col] = pd.to_numeric(density[col], errors="coerce").astype("float64")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    density.to_pickle(cache_path)
    if verbose:
        print(f"  Cached {len(density):,} food codes x {len(density.columns) - 1} components")
    return density


def build_food_search_index(
    drxfcd_path: Path,
    density_df: pd.DataFrame,
    cache_path: Path,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build (or load cached) NHANES food code -> description search index.

    Args:
        drxfcd_path: Path to NHANES P_DRXFCD.XPT (food code dictionary)
        density_df: DataFrame from build_nutrient_density_cache(), used to
                    filter to only foods we have nutrient data for
        cache_path: Where to read/write the pickled cache
        force_rebuild: If True, ignore existing cache and rebuild
        verbose: Print progress messages

    Returns:
        DataFrame with columns: food_code (int64), description (str),
        search_text (str, normalize_text(description)).
    """
    if cache_path.exists() and not force_rebuild:
        if verbose:
            print(f"Food index cache exists: {cache_path}")
        return pd.read_pickle(cache_path)

    if verbose:
        print("Building food code search index...")

    df_fcd = pd.read_sas(drxfcd_path, format="xport")

    # Prefer long description; fall back to short
    desc_col = "DRXFCLD" if "DRXFCLD" in df_fcd.columns else "DRXFCSD"
    df_fcd["description"] = df_fcd[desc_col].apply(
        lambda x: x.decode("latin-1").strip() if isinstance(x, bytes) else str(x).strip()
    )
    df_fcd["food_code"] = df_fcd["DRXFDCD"].astype(int)
    df_fcd["search_text"] = df_fcd["description"].apply(normalize_text)

    index = df_fcd[["food_code", "description", "search_text"]].copy()

    # Filter: only keep foods we have nutrient data for
    index = index[index["food_code"].isin(density_df["food_code"])].copy()

    index["food_code"] = index["food_code"].astype("int64")
    index["description"] = index["description"].astype(str)
    index["search_text"] = index["search_text"].astype(str)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_pickle(cache_path)
    if verbose:
        print(f"  Indexed {len(index):,} food codes")
    return index


def load_nutrient_density_cache(cache_path: Path) -> pd.DataFrame:
    """Load the per-100g nutrient density cache. Raises if missing."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Nutrient density cache not found at {cache_path}. "
            "Build it via dietary.density.build_nutrient_density_cache() "
            "or run `python -m immunosense.agents.dietary.layer1_train`."
        )
    return pd.read_pickle(cache_path)
