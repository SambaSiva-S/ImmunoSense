"""Layer 1 training for Agent 2: NHANES -> DII percentile quantile regressors.

This is a STANDALONE training script that runs once (offline) to produce
the Layer 1 artifacts that runtime code consumes:
    - dii_quantile_10.pkl ... dii_quantile_99.pkl (LightGBM quantile regressors)
    - dii_scoring_reference.pkl (DII_REF + feature ordering)

The Layer 2 nutrient density cache and food search index are also built here.

Usage::

    python -m immunosense.agents.dietary.layer1_train \\
        [--data-dir ./data/nhanes] \\
        [--layer1-dir ./artifacts/agent2_layer1] \\
        [--layer2-dir ./artifacts/agent2_layer2]

Downloads NHANES 2017-March 2020 files (cached), computes DII per participant,
trains quantile regressors on (age, sex, bmi) -> DII at q = {0.10, 0.25, 0.50,
0.75, 0.90, 0.95, 0.99}, and persists artifacts via joblib.

This module has no runtime dependencies on the rest of Agent 2. It uses the
shared constants and DII scoring math but otherwise stands alone.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.dietary.constants import (
    DII_REF,
    LAYER1_FEATURE_COLS,
    MIN_DII_COMPONENTS,
    NHANES_BASE,
    NHANES_DR1TOT_COL_MAP,
    NHANES_FILES,
    QUANTILES,
)
from immunosense.agents.dietary.density import (
    build_food_search_index,
    build_nutrient_density_cache,
)
from immunosense.agents.dietary.dii import compute_dii_row


def download_nhanes(data_dir: Path, verbose: bool = True) -> dict:
    """Download NHANES 2017-Mar-2020 data files (idempotent, cached)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, description in NHANES_FILES.items():
        target = data_dir / filename
        paths[filename] = target
        if target.exists() and target.stat().st_size > 0:
            if verbose:
                print(f"  cached: {filename}")
            continue
        url = f"{NHANES_BASE}/{filename}"
        try:
            urllib.request.urlretrieve(url, target)
            size_kb = target.stat().st_size // 1024
            if verbose:
                print(f"  downloaded: {filename} ({size_kb} KB)")
        except Exception as e:
            if verbose:
                print(f"  FAILED: {filename}: {e}")
            if target.exists():
                target.unlink()
    return paths


def build_dii_features(df_dr1tot: pd.DataFrame) -> pd.DataFrame:
    """Build the DII feature matrix from NHANES DR1TOT.

    Args:
        df_dr1tot: NHANES P_DR1TOT DataFrame.

    Returns:
        DataFrame with columns: SEQN, plus one column per DII component
        with intake values (or NaN if column missing).
    """
    dii_features = pd.DataFrame({"SEQN": df_dr1tot["SEQN"]})

    for component, nhanes_col in NHANES_DR1TOT_COL_MAP.items():
        if nhanes_col is None:
            continue
        if nhanes_col in df_dr1tot.columns:
            dii_features[component] = df_dr1tot[nhanes_col]
        else:
            dii_features[component] = np.nan

    # Caffeine: NHANES reports mg, DII_REF expects g
    if "caffeine_g" in dii_features.columns:
        dii_features["caffeine_g"] = dii_features["caffeine_g"] / 1000.0

    # Omega-3 / Omega-6 from individual fatty acid columns
    omega3_cols = [
        c for c in ["DR1TP183", "DR1TP205", "DR1TP225", "DR1TP226"]
        if c in df_dr1tot.columns
    ]
    omega6_cols = [
        c for c in ["DR1TP182", "DR1TP204"] if c in df_dr1tot.columns
    ]
    if omega3_cols:
        dii_features["omega3_g"] = df_dr1tot[omega3_cols].sum(axis=1)
    if omega6_cols:
        dii_features["omega6_g"] = df_dr1tot[omega6_cols].sum(axis=1)

    return dii_features


def train_quantile_regressors(
    df: pd.DataFrame,
    feature_cols: list = LAYER1_FEATURE_COLS,
    quantiles: list = QUANTILES,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train LightGBM quantile regressors on (age, sex, bmi) -> dii_score.

    Args:
        df: DataFrame with columns 'age', 'sex', 'bmi', 'dii_score'.
        feature_cols: Demographic feature columns.
        quantiles: List of quantiles to train regressors for.
        random_state: Train/test split seed and LightGBM seed.
        verbose: Print empirical coverage per quantile.

    Returns:
        Dict {quantile: trained_model}.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X = df[feature_cols].astype(float)
    y = df["dii_score"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    dii_models = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=15,
            min_child_samples=50,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        dii_models[q] = model
        if verbose:
            coverage = (y_test <= model.predict(X_test)).mean()
            print(f"  q={q:.2f}: empirical coverage = {coverage:.3f} (target {q:.3f})")

    return dii_models


def run_layer1_training(
    data_dir: Path,
    layer1_dir: Path,
    layer2_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Run the full Layer 1 (and optionally Layer 2 cache build) training pipeline.

    Args:
        data_dir: NHANES data directory (will download if files missing).
        layer1_dir: Where to write trained quantile models + scoring reference.
        layer2_dir: Where to write nutrient density cache + food search index
                     (optional; if None, Layer 2 caches are not built).
        verbose: Print progress messages.

    Returns:
        Dict with keys: 'dii_models', 'dii_ref', 'feature_cols'
                       (and 'density_cache_path', 'food_index_path' if
                        layer2_dir was provided)
    """
    import joblib

    if verbose:
        print("=" * 70)
        print("Agent 2 Layer 1 training")
        print("=" * 70)

    # 1. Download
    if verbose:
        print("\n[1/5] Downloading NHANES data...")
    paths = download_nhanes(data_dir, verbose=verbose)

    # 2. Load
    if verbose:
        print("\n[2/5] Loading NHANES tables...")
    df_dr1tot = pd.read_sas(paths["P_DR1TOT.XPT"], format="xport")
    df_crp = pd.read_sas(paths["P_HSCRP.XPT"], format="xport")
    df_demo = pd.read_sas(paths["P_DEMO.XPT"], format="xport")
    df_bmx = pd.read_sas(paths["P_BMX.XPT"], format="xport")

    # Filter to reliable dietary recalls
    if "DR1DRSTZ" in df_dr1tot.columns:
        df_dr1tot = df_dr1tot[df_dr1tot["DR1DRSTZ"] == 1].copy()

    # Select demographic columns
    df_demo = df_demo[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]].rename(
        columns={"RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH3": "race_eth"}
    )
    df_bmx = df_bmx[["SEQN", "BMXBMI"]].rename(columns={"BMXBMI": "bmi"})
    df_crp = df_crp[["SEQN", "LBXHSCRP"]].rename(columns={"LBXHSCRP": "crp_mg_l"})

    # 3. Compute DII per participant
    if verbose:
        print("\n[3/5] Computing DII per participant...")
    dii_features = build_dii_features(df_dr1tot)

    # Filter to participants with enough non-NaN components
    components_per_row = dii_features.drop(columns="SEQN").notna().sum(axis=1)
    dii_features = dii_features[components_per_row >= MIN_DII_COMPONENTS].copy()
    if verbose:
        print(f"  After >={MIN_DII_COMPONENTS} components filter: "
              f"{len(dii_features):,} participants")

    dii_scores = dii_features.apply(lambda r: compute_dii_row(r, DII_REF), axis=1)
    df_dii = pd.DataFrame({
        "SEQN": dii_features["SEQN"],
        "dii_score": dii_scores,
    })

    # Merge to adult complete-case dataset
    df = (df_dii
          .merge(df_demo, on="SEQN", how="inner")
          .merge(df_bmx, on="SEQN", how="inner")
          .merge(df_crp, on="SEQN", how="inner"))
    df = df.dropna(subset=["dii_score", "age", "sex", "bmi", "crp_mg_l"])
    df = df[df["age"] >= 18].copy()
    if verbose:
        print(f"  Adult complete cases: {len(df):,}")

    # 4. Train quantile regressors
    if verbose:
        print("\n[4/5] Training quantile regressors...")
    dii_models = train_quantile_regressors(df, verbose=verbose)

    # 5. Persist Layer 1 artifacts
    if verbose:
        print(f"\n[5/5] Persisting artifacts to {layer1_dir}/")
    layer1_dir.mkdir(parents=True, exist_ok=True)
    for q, model in dii_models.items():
        joblib.dump(model, layer1_dir / f"dii_quantile_{int(q * 100):02d}.pkl")

    joblib.dump(
        {
            "dii_ref": DII_REF,
            "feature_cols": LAYER1_FEATURE_COLS,
        },
        layer1_dir / "dii_scoring_reference.pkl",
    )

    if verbose:
        print(f"  Saved {len(dii_models)} quantile models + scoring reference")

    result = {
        "dii_models": dii_models,
        "dii_ref": DII_REF,
        "feature_cols": LAYER1_FEATURE_COLS,
    }

    # Optional: build Layer 2 caches
    if layer2_dir is not None:
        if verbose:
            print(f"\n[Bonus] Building Layer 2 caches in {layer2_dir}/")
        layer2_dir.mkdir(parents=True, exist_ok=True)
        density_cache = layer2_dir / "nutrient_density_per_100g_v2.pkl"
        food_index_cache = layer2_dir / "food_code_search_index.pkl"

        density_df = build_nutrient_density_cache(
            paths["P_DR1IFF.XPT"], density_cache, verbose=verbose,
        )
        build_food_search_index(
            paths["P_DRXFCD.XPT"], density_df, food_index_cache, verbose=verbose,
        )
        result["density_cache_path"] = density_cache
        result["food_index_path"] = food_index_cache

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Agent 2 Layer 1 (NHANES -> DII quantile regressors)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data/nhanes"),
        help="NHANES data directory (default: ./data/nhanes)",
    )
    parser.add_argument(
        "--layer1-dir", type=Path, default=Path("./artifacts/agent2_layer1"),
        help="Layer 1 artifact directory (default: ./artifacts/agent2_layer1)",
    )
    parser.add_argument(
        "--layer2-dir", type=Path, default=Path("./artifacts/agent2_layer2"),
        help="Layer 2 cache directory (default: ./artifacts/agent2_layer2)",
    )
    parser.add_argument(
        "--no-layer2", action="store_true",
        help="Skip building Layer 2 nutrient density + food index caches",
    )
    args = parser.parse_args()

    run_layer1_training(
        data_dir=args.data_dir,
        layer1_dir=args.layer1_dir,
        layer2_dir=None if args.no_layer2 else args.layer2_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
