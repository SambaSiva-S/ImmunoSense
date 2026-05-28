"""CRP population baseline via LightGBM quantile regression.

Loads trained NHANES quantile models and exposes:
    CRPBaseline.percentile(age, sex, bmi, crp_value) -> float

The percentile mapping uses 7 quantile regressors (q in [0.10, 0.25, 0.50,
0.75, 0.90, 0.95, 0.99]). For an observed CRP, we find the smallest q
such that the predicted q-th quantile threshold >= the observed value.
Values above the 0.99 threshold return 0.99.

For demographic-aware lookup with linear interpolation between quantiles,
see `get_dii_percentile`-style interpolation in agent 2's dii module — this
module uses a simpler step-function lookup matching the source notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.constants import LAYER1_FEATURE_COLS, QUANTILES


class CRPBaseline:
    """Holds the trained quantile models for population CRP percentile lookup.

    Args:
        models: dict {quantile: trained_model}. Each model.predict(X) returns
            the predicted CRP at that quantile for one row.
        feature_cols: List of feature columns expected at prediction time.
            Defaults to LAYER1_FEATURE_COLS = ["age", "sex", "bmi"].

    Example:
        >>> baseline = CRPBaseline.load(Path("./artifacts/agent1_layer1"))
        >>> baseline.percentile(age=45, sex=2, bmi=28, crp_value=8.0)
        0.95
    """

    def __init__(
        self,
        models: dict,
        feature_cols: list = LAYER1_FEATURE_COLS,
    ) -> None:
        self.models = models
        self.feature_cols = feature_cols
        # Cache the sorted quantile list for deterministic iteration
        self._sorted_quantiles = sorted(models.keys())

    @classmethod
    def load(cls, artifact_dir: Path) -> "CRPBaseline":
        """Load all quantile models from disk.

        Expects files like `crp_quantile_10.pkl`, `crp_quantile_25.pkl`, etc.
        """
        import joblib

        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Layer 1 artifacts directory not found: {artifact_dir}. "
                "Train via `python -m immunosense.agents.biomarker.layer1.train`."
            )

        models = {}
        for q in QUANTILES:
            path = artifact_dir / f"crp_quantile_{int(q * 100):02d}.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Quantile model missing: {path}. Re-train Layer 1."
                )
            models[q] = joblib.load(path)

        # Load metadata if present
        meta_path = artifact_dir / "layer1_metadata.pkl"
        feature_cols = LAYER1_FEATURE_COLS
        if meta_path.exists():
            meta = joblib.load(meta_path)
            feature_cols = meta.get("feature_cols", LAYER1_FEATURE_COLS)

        return cls(models=models, feature_cols=feature_cols)

    def predict_quantiles(self, age: float, sex: int, bmi: float) -> dict:
        """Predict CRP at each trained quantile for one demographic."""
        X = pd.DataFrame([[age, sex, bmi]], columns=self.feature_cols)
        return {q: float(m.predict(X)[0]) for q, m in self.models.items()}

    def percentile(
        self, age: float, sex: int, bmi: float, crp_value: float,
    ) -> float:
        """Given demographics and a CRP value, return the percentile.

        Uses step-function lookup: returns smallest q such that the predicted
        q-th quantile >= crp_value. Returns 0.99 if crp_value exceeds all
        trained quantiles.

        Args:
            age: years
            sex: NHANES convention (1=male, 2=female)
            bmi: kg/m^2
            crp_value: mg/L

        Returns:
            Percentile in [smallest_quantile, 0.99].
        """
        X = pd.DataFrame([[age, sex, bmi]], columns=self.feature_cols)
        for q in self._sorted_quantiles:
            threshold = float(self.models[q].predict(X)[0])
            if crp_value <= threshold:
                return q
        return 0.99


def get_crp_percentile(
    age: float,
    sex: int,
    bmi: float,
    crp_value: float,
    baseline: CRPBaseline,
) -> float:
    """Convenience wrapper around CRPBaseline.percentile().

    Args:
        baseline: A pre-loaded CRPBaseline. Required; this is the stateless
            entry point that mirrors the function name from the original
            notebook for ergonomics.
    """
    return baseline.percentile(age, sex, bmi, crp_value)


def load_layer1_artifacts(artifact_dir: Path) -> CRPBaseline:
    """Load the Layer 1 trained CRP baseline from artifact_dir.

    Convenience function that returns a CRPBaseline. Equivalent to
    CRPBaseline.load(artifact_dir).
    """
    return CRPBaseline.load(artifact_dir)
