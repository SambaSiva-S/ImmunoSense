"""Layer 2: Disease intelligence via three cognitive pillars.

    Pillar A (Spatial):     DiseaseEncoder + NT-Xent contrastive loss
                            -> 128-dim embedding -> nearest-centroid classification
    Pillar B (Probabilistic): LightGBM multiclass classifier with class_weight='balanced'
    Pillar C (Explanatory):   XGBoost classifier + SHAP TreeExplainer

Fusion: weighted average of the three pillars' class probabilities with
softmax-normalized cosine similarities from Pillar A.

Trained offline via:
    python -m immunosense.agents.biomarker.layer2.train

Runtime loads all artifacts via Layer2Bundle.load(layer2_dir).
"""

from immunosense.agents.biomarker.layer2.classifier import train_lgb_classifier
from immunosense.agents.biomarker.layer2.encoder import (
    DiseaseEncoder,
    NTXentLoss,
    compute_centroids,
    train_contrastive_encoder,
)
from immunosense.agents.biomarker.layer2.explainer import (
    compute_shap_values,
    top_shap_drivers,
    train_xgb_classifier,
)
from immunosense.agents.biomarker.layer2.fusion import fuse_predictions
from immunosense.agents.biomarker.layer2.pipeline import Layer2Bundle

__all__ = [
    "DiseaseEncoder",
    "NTXentLoss",
    "compute_centroids",
    "train_contrastive_encoder",
    "train_lgb_classifier",
    "compute_shap_values",
    "top_shap_drivers",
    "train_xgb_classifier",
    "fuse_predictions",
    "Layer2Bundle",
]
