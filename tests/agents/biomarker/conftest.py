"""Shared fixtures for biomarker tests.

Builds a small in-memory synthetic Rheumatic dataset and a small NHANES-shaped
dataset, then trains tiny Layer 1 and Layer 2 artifacts once per test session.
Slow fixtures are session-scoped to keep the test suite under ~30s.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Tiny synthetic Rheumatic-shaped dataset
# ============================================================

@pytest.fixture(scope="session")
def synthetic_rheumatic_df() -> pd.DataFrame:
    """Build a 1400-row synthetic Rheumatic dataset (200 per class)."""
    np.random.seed(42)
    rows = []
    disease_means = {
        "Ankylosing Spondylitis":      dict(Age=42, ESR=35, CRP=12, RF=15, AntiCCP=8, C3=110, C4=24, HLAB27=0.95),
        "Normal":                       dict(Age=40, ESR=10, CRP=1.5, RF=8, AntiCCP=3, C3=125, C4=28, HLAB27=0.05),
        "Psoriatic Arthritis":          dict(Age=45, ESR=25, CRP=8, RF=20, AntiCCP=10, C3=115, C4=25, HLAB27=0.25),
        "Rheumatoid Arthritis":         dict(Age=52, ESR=42, CRP=22, RF=85, AntiCCP=90, C3=108, C4=23, HLAB27=0.05),
        "Sjogren's Syndrome":           dict(Age=55, ESR=30, CRP=5, RF=40, AntiCCP=15, C3=100, C4=20, HLAB27=0.05),
        "Systemic Lupus Erythematosus": dict(Age=38, ESR=48, CRP=15, RF=25, AntiCCP=20, C3=75, C4=14, HLAB27=0.05),
        "Systemic Sclerosis":           dict(Age=50, ESR=30, CRP=8, RF=15, AntiCCP=10, C3=105, C4=22, HLAB27=0.05),
    }

    for disease, means in disease_means.items():
        for _ in range(200):
            row = {"Disease": disease}
            row["Age"] = max(18, np.random.normal(means["Age"], 10))
            row["ESR"] = max(0, np.random.normal(means["ESR"], 12))
            row["CRP"] = max(0, np.random.normal(means["CRP"], 5))
            row["RF"] = max(0, np.random.normal(means["RF"], 25))
            row["Anti-CCP"] = max(0, np.random.normal(means["AntiCCP"], 25))
            row["C3"] = max(50, np.random.normal(means["C3"], 15))
            row["C4"] = max(5, np.random.normal(means["C4"], 5))
            row["Gender"] = np.random.choice(["Male", "Female"])
            row["HLA-B27"] = "Positive" if np.random.rand() < means["HLAB27"] else "Negative"
            ana_pos = (disease == "Systemic Lupus Erythematosus" and np.random.rand() < 0.95) or np.random.rand() < 0.1
            row["ANA"] = "Positive" if ana_pos else "Negative"
            for col in ["Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]:
                sle_pos = disease == "Systemic Lupus Erythematosus" and np.random.rand() < 0.4
                row[col] = "Positive" if sle_pos else "Negative"
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_rheumatic_xlsx(synthetic_rheumatic_df: pd.DataFrame, tmp_path_factory) -> Path:
    """Write the synthetic Rheumatic dataset to a temp .xlsx file."""
    tmpdir = tmp_path_factory.mktemp("rheumatic")
    xlsx = tmpdir / "synthetic_rheumatic.xlsx"
    synthetic_rheumatic_df.to_excel(xlsx, index=False)
    return xlsx


# ============================================================
# Trained Layer 2 bundle (slow, session-scoped)
# ============================================================

@pytest.fixture(scope="session")
def trained_layer2_dir(synthetic_rheumatic_xlsx: Path, tmp_path_factory) -> Path:
    """Train a tiny Layer 2 (10 encoder epochs) and return its artifact dir."""
    from immunosense.agents.biomarker.layer2.train import run_layer2_training
    layer2_dir = tmp_path_factory.mktemp("layer2") / "agent1_layer2"
    run_layer2_training(
        rheumatic_xlsx=synthetic_rheumatic_xlsx,
        artifact_dir=layer2_dir,
        encoder_epochs=10,  # quick
        verbose=False,
    )
    return layer2_dir


@pytest.fixture(scope="session")
def loaded_layer2_bundle(trained_layer2_dir: Path):
    """Load the Layer 2 bundle once for tests that don't re-train."""
    from immunosense.agents.biomarker.layer2 import Layer2Bundle
    return Layer2Bundle.load(trained_layer2_dir)


# ============================================================
# A typical "test reading" (RA-shaped)
# ============================================================

@pytest.fixture
def ra_reading() -> dict:
    return {
        "day": 0,
        "Age": 52,
        "ESR": 42,
        "CRP": 22.0,
        "RF": 85.0,
        "Anti-CCP": 90.0,
        "C3": 108.0,
        "C4": 23.0,
        "Gender_enc": 1,
        "HLA-B27_enc": 0,
        "ANA_enc": 0,
        "Anti-Ro_enc": 0,
        "Anti-La_enc": 0,
        "Anti-dsDNA_enc": 0,
        "Anti-Sm_enc": 0,
        "gluten_exposure": False,
        "poor_sleep": False,
        "high_stress": False,
        "high_aqi": False,
    }


@pytest.fixture
def normal_reading() -> dict:
    return {
        "day": 0,
        "Age": 30,
        "ESR": 8,
        "CRP": 1.0,
        "RF": 5.0,
        "Anti-CCP": 2.0,
        "C3": 125.0,
        "C4": 28.0,
        "Gender_enc": 0,
        "HLA-B27_enc": 0,
        "ANA_enc": 0,
        "Anti-Ro_enc": 0,
        "Anti-La_enc": 0,
        "Anti-dsDNA_enc": 0,
        "Anti-Sm_enc": 0,
    }
