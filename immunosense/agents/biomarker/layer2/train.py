"""Layer 2 training: Rheumatic dataset -> 3 cognitive pillars.

Standalone training script. Run once (offline) to produce all 6 Layer 2
artifacts in one shot:
    lgb_model.pkl       (Pillar B LightGBM)
    xgb_model.json      (Pillar C XGBoost)
    encoder.pt          (Pillar A contrastive encoder weights)
    centroids.npz       (per-class centroids on unit sphere)
    scaler.pkl          (StandardScaler for encoder input)
    label_encoder.pkl   (sklearn LabelEncoder)
    layer2_metadata.pkl (input_dim, embedding_dim, feature_names)

Usage::

    python -m immunosense.agents.biomarker.layer2.train \\
        --rheumatic-xlsx ./data/rheumatic/Rheumatic_and_Autoimmune_Disease_Dataset.xlsx \\
        [--artifact-dir ./artifacts/agent1_layer2]

Requires: pandas, numpy, scikit-learn, lightgbm, xgboost, shap, torch, openpyxl.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.constants import (
    ALL_INPUT_FEATURES,
    ALL_VALUE_FEATURES,
    LAYER2_EMBEDDING_DIM,
    NUMERIC_BIOMARKER_FEATURES,
)
from immunosense.agents.biomarker.layer2.classifier import train_lgb_classifier
from immunosense.agents.biomarker.layer2.encoder import (
    compute_centroids,
    train_contrastive_encoder,
)
from immunosense.agents.biomarker.layer2.explainer import train_xgb_classifier
from immunosense.agents.biomarker.layer2.pipeline import Layer2Bundle


def prepare_rheumatic_dataset(xlsx_path: Path, verbose: bool = True) -> tuple:
    """Load and prepare the Rheumatic dataset for Layer 2 training.

    Encodes categorical features (Positive/Negative -> 1/0), creates missing
    indicators, and returns the final (X, y, label_encoder) tuple.

    Returns:
        Tuple (X, y, label_encoder, feature_names, class_names).
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_excel(xlsx_path)

    if verbose:
        print(f"  Rheumatic dataset shape: {df.shape}")
        print(f"  Disease distribution:")
        for d, n in df["Disease"].value_counts().items():
            print(f"    {d}: {n}")

    # Encode categorical features
    df["Gender_enc"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["HLA-B27_enc"] = df["HLA-B27"].map({"Positive": 1, "Negative": 0})
    df["ANA_enc"] = df["ANA"].map({"Positive": 1, "Negative": 0})
    for col in ["Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]:
        df[f"{col}_enc"] = df[col].map({"Positive": 1, "Negative": 0})

    # Create missing indicators
    for feat in ALL_VALUE_FEATURES:
        df[f"{feat}_missing"] = df[feat].isnull().astype(int)

    # Fill NaN with 0 (missing indicator captures the info)
    X = df[ALL_INPUT_FEATURES].fillna(0).values.astype(float)

    le = LabelEncoder()
    y = le.fit_transform(df["Disease"])
    class_names = list(le.classes_)

    if verbose:
        print(f"  Features: {len(ALL_INPUT_FEATURES)} ({len(ALL_VALUE_FEATURES)} values + {len(ALL_VALUE_FEATURES)} missing flags)")
        print(f"  Classes: {class_names}")
        print(f"  Samples: {len(X)}")

    return X, y, le, ALL_INPUT_FEATURES, class_names


def run_layer2_training(
    rheumatic_xlsx: Path,
    artifact_dir: Path,
    embedding_dim: int = LAYER2_EMBEDDING_DIM,
    encoder_epochs: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> Layer2Bundle:
    """Run the full Layer 2 training pipeline.

    Args:
        rheumatic_xlsx: path to the Rheumatic dataset .xlsx
        artifact_dir: where to write the 6 trained artifacts
        embedding_dim: contrastive encoder output dimension
        encoder_epochs: training epochs for Pillar A
        random_state: seed for split/training
        verbose: print progress

    Returns:
        A loaded Layer2Bundle (also persisted to disk).
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if verbose:
        print("=" * 70)
        print("Agent 1 Layer 2 training (3 cognitive pillars)")
        print("=" * 70)

    # 1. Load + prepare data
    if verbose:
        print("\n[1/6] Loading Rheumatic dataset...")
    X, y, le, feature_names, class_names = prepare_rheumatic_dataset(
        rheumatic_xlsx, verbose=verbose,
    )

    # 2. Split: 80% train, 10% val, 10% test
    if verbose:
        print("\n[2/6] Splitting into train/val/test (80/10/10)...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.111,  # 0.111 of 0.9 -> ~0.10 of total
        random_state=random_state, stratify=y_trainval,
    )
    if verbose:
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 3. Scale features (for encoder; tree models don't need it)
    # Use DataFrames so feature names propagate to LGB/XGB classifiers,
    # which silences sklearn's "X does not have valid feature names" warning
    # at inference time.
    X_train_df = pd.DataFrame(X_train, columns=ALL_INPUT_FEATURES)
    X_val_df = pd.DataFrame(X_val, columns=ALL_INPUT_FEATURES)
    X_test_df = pd.DataFrame(X_test, columns=ALL_INPUT_FEATURES)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 4. Pillar B: LightGBM
    if verbose:
        print("\n[3/6] Training Pillar B (LightGBM classifier)...")
    lgb_model = train_lgb_classifier(
        X_train_df, y_train, X_val_df, y_val,
        random_state=random_state, verbose=verbose,
    )

    # 5. Pillar C: XGBoost
    if verbose:
        print("\n[4/6] Training Pillar C (XGBoost classifier + SHAP-ready)...")
    xgb_model = train_xgb_classifier(
        X_train_df, y_train, n_classes=len(class_names),
        X_val=X_val_df, y_val=y_val,
        random_state=random_state, verbose=verbose,
    )

    # 6. Pillar A: Contrastive encoder
    if verbose:
        print("\n[5/6] Training Pillar A (contrastive encoder)...")
    encoder, losses = train_contrastive_encoder(
        X_train_scaled, y_train,
        embedding_dim=embedding_dim,
        epochs=encoder_epochs,
        random_seed=random_state,
        verbose=verbose,
    )

    if verbose:
        print("  Computing disease centroids...")
    centroids = compute_centroids(encoder, X_train_scaled, y_train, class_names)
    if verbose:
        print(f"  Built {len(centroids)} centroids on unit sphere")
        print(f"  Encoder loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # 7. Quick test-set evaluation (sanity check)
    if verbose:
        print("\n  Evaluating individual pillars on test set...")
    lgb_acc = accuracy_score(y_test, lgb_model.predict(X_test_df))
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_df))
    if verbose:
        print(f"    Pillar B (LightGBM) test: {lgb_acc * 100:.1f}%")
        print(f"    Pillar C (XGBoost)  test: {xgb_acc * 100:.1f}%")

    # 8. Persist artifacts
    if verbose:
        print(f"\n[6/6] Persisting artifacts to {artifact_dir}/")
    bundle = Layer2Bundle(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        encoder=encoder,
        centroids=centroids,
        scaler=scaler,
        label_encoder=le,
        feature_names=feature_names,
        class_names=class_names,
    )
    bundle.save(artifact_dir)

    if verbose:
        print("  Saved: lgb_model.pkl, xgb_model.json, encoder.pt, centroids.npz,")
        print("         scaler.pkl, label_encoder.pkl, layer2_metadata.pkl")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Agent 1 Layer 2 (3 cognitive pillars)"
    )
    parser.add_argument(
        "--rheumatic-xlsx", type=Path, required=True,
        help="Path to the Rheumatic and Autoimmune Disease dataset .xlsx",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("./artifacts/agent1_layer2"),
        help="Layer 2 artifact directory",
    )
    parser.add_argument(
        "--encoder-epochs", type=int, default=100,
        help="Training epochs for the contrastive encoder",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    run_layer2_training(
        rheumatic_xlsx=args.rheumatic_xlsx,
        artifact_dir=args.artifact_dir,
        encoder_epochs=args.encoder_epochs,
        random_state=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
