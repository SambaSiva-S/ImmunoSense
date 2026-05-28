"""Layer 1: NHANES-based population CRP baseline.

LightGBM quantile regression on (age, sex, BMI) -> CRP distribution.
Trained offline via:
    python -m immunosense.agents.biomarker.layer1.train

Runtime loads the pickled artifacts via load_layer1_artifacts().
"""

from immunosense.agents.biomarker.layer1.crp_baseline import (
    CRPBaseline,
    get_crp_percentile,
    load_layer1_artifacts,
)

__all__ = [
    "CRPBaseline",
    "get_crp_percentile",
    "load_layer1_artifacts",
]
