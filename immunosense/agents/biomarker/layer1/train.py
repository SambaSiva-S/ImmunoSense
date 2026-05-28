"""Layer 1 training: NHANES -> CRP quantile regression models.

Standalone training script. Run once (offline) to produce:
    crp_quantile_10.pkl ... crp_quantile_99.pkl  (7 LightGBM quantile models)
    layer1_metadata.pkl                            (feature ordering)

Usage::

    python -m immunosense.agents.biomarker.layer1.train \\
        [--data-dir ./data/nhanes] \\
        [--artifact-dir ./artifacts/agent1_layer1]

Downloads NHANES 2017-March 2020 (CRP + demographics + BMI + CBC), filters
to complete adult cases with valid CRP, trains 7 LightGBM quantile
regressors on (age, sex, BMI) -> CRP, and persists artifacts via joblib.

This module has no runtime dependency on the rest of Agent 1 beyond
constants. It does require lightgbm, scikit-learn, and pyreadstat
(for NHANES .XPT files).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.constants import (
    LAYER1_FEATURE_COLS,
    NHANES_BASE,
    NHANES_FILES,
    QUANTILES,
)


def download_nhanes(data_dir: Path, verbose: bool = True) -> dict:
    """Download NHANES 2017-March 2020 files (idempotent, cached).

    Note: Agent 1's source notebook used filenames like 'HSCRP_L.XPT',
    'DEMO_L.XPT' etc. (cycle L). We use 'P_*' (pre-pandemic) for
    consistency with Agent 2's Layer 1 training, which uses the same
    NHANES cycle and gives us the same demographic universe.
    """
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


def prepare_nhanes_dataset(paths: dict, verbose: bool = True) -> pd.DataFrame:
    """Load and merge NHANES tables into one adult complete-case DataFrame.

    Source-faithful join pattern (matches 01_Biomarker_Exploration.py L68-72):
        df_crp INNER df_demo  (require demographics)
        +     LEFT  df_bmx   (BMI)
        +     LEFT  df_cbc   (complete blood count: WBC, Lymphocyte_pct, Neutrophil_pct, Platelets)

    The CBC columns are joined as LEFT so they can be NaN, but the subsequent
    filter requires WBC presence — this implicitly excludes ~5-10% of NHANES
    participants who skipped the CBC blood draw, ensuring our training
    population is "patients who had a full blood workup". This matches the
    source notebook's filter at L251: df[['Age','Sex','BMI','CRP','WBC']].dropna().

    The CBC columns (WBC, Lymphocyte_pct, Neutrophil_pct, Platelets) are kept
    in the returned DataFrame for reference but are NOT used as model features
    — Layer 1 trains only on (Age, Sex, BMI) -> CRP, matching the source's
    feature selection at L258.

    Returns:
        DataFrame with columns: SEQN, Age, Sex, BMI, CRP, WBC, Lymphocyte_pct,
        Neutrophil_pct, Platelets. Filtered to adults (age >= 18) with non-null
        Age/Sex/BMI/CRP/WBC, positive CRP, and CRP < 50 (extreme outlier removal
        matching the notebook).
    """
    df_crp = pd.read_sas(paths["P_HSCRP.XPT"], format="xport")
    df_demo = pd.read_sas(paths["P_DEMO.XPT"], format="xport")
    df_bmx = pd.read_sas(paths["P_BMX.XPT"], format="xport")
    df_cbc = pd.read_sas(paths["P_CBC.XPT"], format="xport")

    if verbose:
        print(f"  HSCRP rows: {len(df_crp):,}")
        print(f"  DEMO rows:  {len(df_demo):,}")
        print(f"  BMX rows:   {len(df_bmx):,}")
        print(f"  CBC rows:   {len(df_cbc):,}")

    df = (
        df_crp[["SEQN", "LBXHSCRP"]]
        .rename(columns={"LBXHSCRP": "CRP"})
        .merge(
            df_demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].rename(
                columns={"RIDAGEYR": "Age", "RIAGENDR": "Sex"}
            ),
            on="SEQN",
            how="inner",
        )
        .merge(
            df_bmx[["SEQN", "BMXBMI"]].rename(columns={"BMXBMI": "BMI"}),
            on="SEQN",
            how="left",
        )
        .merge(
            df_cbc[["SEQN", "LBXWBCSI", "LBXLYPCT", "LBXNEPCT", "LBXPLTSI"]].rename(
                columns={
                    "LBXWBCSI": "WBC",
                    "LBXLYPCT": "Lymphocyte_pct",
                    "LBXNEPCT": "Neutrophil_pct",
                    "LBXPLTSI": "Platelets",
                }
            ),
            on="SEQN",
            how="left",
        )
    )

    # Filter: complete cases requiring WBC presence (source L251), adult,
    # positive CRP, non-extreme. Note: WBC is in the dropna columns but NOT
    # used as a model feature — it just enforces "patient had CBC done".
    df = df.dropna(subset=["Age", "Sex", "BMI", "CRP", "WBC"])
    df = df[df["Age"] >= 18]
    df = df[df["CRP"] > 0]
    df = df[df["CRP"] < 50]

    if verbose:
        print(f"  Adult complete cases (CRP in (0,50), WBC present): {len(df):,}")

    return df.reset_index(drop=True)


def train_quantile_regressors(
    df: pd.DataFrame,
    feature_cols: list = LAYER1_FEATURE_COLS,
    quantiles: list = QUANTILES,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train 7 LightGBM quantile regressors on (age, sex, BMI) -> CRP.

    Returns:
        dict {quantile: trained_LGBMRegressor}.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    # The dataframe has lowercase or uppercase column names depending on caller;
    # be defensive and try both.
    actual_features = []
    for col in feature_cols:
        if col in df.columns:
            actual_features.append(col)
        elif col.capitalize() in df.columns:
            actual_features.append(col.capitalize())
        else:
            raise KeyError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    X = df[actual_features].astype(float).values
    y = df["CRP"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    models = {}
    if verbose:
        print(f"  Training {len(quantiles)} quantile models on {len(X_train):,} samples...")
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        models[q] = model
        if verbose:
            preds = model.predict(X_test)
            coverage = float((y_test <= preds).mean())
            print(
                f"    q={q:.2f}: coverage={coverage:.3f} (target {q:.3f}), "
                f"mean_pred={preds.mean():.2f}"
            )

    return models


def run_layer1_training(
    data_dir: Path,
    artifact_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run the full Layer 1 training pipeline end-to-end."""
    import joblib

    if verbose:
        print("=" * 70)
        print("Agent 1 Layer 1 training (CRP quantile regression)")
        print("=" * 70)

    if verbose:
        print("\n[1/4] Downloading NHANES data...")
    paths = download_nhanes(data_dir, verbose=verbose)

    if verbose:
        print("\n[2/4] Preparing dataset...")
    df = prepare_nhanes_dataset(paths, verbose=verbose)

    if verbose:
        print("\n[3/4] Training quantile regressors...")
    models = train_quantile_regressors(df, verbose=verbose)

    if verbose:
        print(f"\n[4/4] Persisting artifacts to {artifact_dir}/")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for q, model in models.items():
        joblib.dump(model, artifact_dir / f"crp_quantile_{int(q * 100):02d}.pkl")
    joblib.dump(
        {"feature_cols": LAYER1_FEATURE_COLS, "n_train_samples": len(df)},
        artifact_dir / "layer1_metadata.pkl",
    )

    if verbose:
        print(f"  Saved {len(models)} quantile models + metadata")

    return {"models": models, "n_samples": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Agent 1 Layer 1 (NHANES -> CRP quantile regressors)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("./data/nhanes"),
        help="NHANES data directory",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("./artifacts/agent1_layer1"),
        help="Layer 1 artifact directory",
    )
    args = parser.parse_args()

    run_layer1_training(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
