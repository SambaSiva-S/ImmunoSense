"""Fusion: combine the 3 Layer 2 pillars into one prediction.

Fusion formula (from source notebook):
    sim_probs = softmax(similarities * 5.0)             # Pillar A as probability distribution
    fused    = 0.30 * sim_probs + 0.35 * P_B + 0.35 * P_C
    fused    = fused / fused.sum()                       # renormalize

This gives Pillar B (LightGBM) and Pillar C (XGBoost) equal weight, with
the contrastive encoder (Pillar A) acting as a complementary spatial signal.
"""

from __future__ import annotations

import numpy as np

from immunosense.agents.biomarker.constants import LAYER2_HYPERPARAMS


def fuse_predictions(
    pillar_a_sims: dict,
    pillar_b_probs: np.ndarray,
    pillar_c_probs: np.ndarray,
    class_names: list,
    weight_a: float = None,
    weight_b: float = None,
    weight_c: float = None,
    similarity_temperature: float = None,
) -> tuple:
    """Fuse the 3 pillars into one prediction.

    Args:
        pillar_a_sims: dict {class_name: cosine_similarity in [-1, 1]}
        pillar_b_probs: np.ndarray (n_classes,) from LightGBM predict_proba
        pillar_c_probs: np.ndarray (n_classes,) from XGBoost predict_proba
        class_names: ordered list of class names (must match prob arrays)
        weight_a, weight_b, weight_c: fusion weights (default 0.30/0.35/0.35)
        similarity_temperature: multiplier on sims before softmax (default 5.0)

    Returns:
        Tuple of:
            pred_idx (int): argmax class index
            confidence (float): fused probability of predicted class
            agreement (bool): True if all 3 pillars picked the same class
            fused_probs (np.ndarray): full fused probability distribution
            sim_probs (np.ndarray): Pillar A as softmax-ed distribution
    """
    weight_a = weight_a if weight_a is not None else LAYER2_HYPERPARAMS["fusion_weight_a"]
    weight_b = weight_b if weight_b is not None else LAYER2_HYPERPARAMS["fusion_weight_b"]
    weight_c = weight_c if weight_c is not None else LAYER2_HYPERPARAMS["fusion_weight_c"]
    similarity_temperature = (
        similarity_temperature
        if similarity_temperature is not None
        else LAYER2_HYPERPARAMS["fusion_similarity_temperature"]
    )

    # Pillar A: turn cosine sims into a probability distribution via softmax
    sim_values = np.array([pillar_a_sims[c] for c in class_names])
    # Numerically stable softmax
    scaled = sim_values * similarity_temperature
    scaled = scaled - scaled.max()
    exp_scaled = np.exp(scaled)
    sim_probs = exp_scaled / exp_scaled.sum()

    pillar_b_probs = np.asarray(pillar_b_probs)
    pillar_c_probs = np.asarray(pillar_c_probs)

    # Weighted fusion
    fused = weight_a * sim_probs + weight_b * pillar_b_probs + weight_c * pillar_c_probs
    total = fused.sum()
    if total > 1e-12:
        fused = fused / total

    pred_idx = int(np.argmax(fused))
    confidence = float(fused[pred_idx])

    a_pred = int(np.argmax(sim_probs))
    b_pred = int(np.argmax(pillar_b_probs))
    c_pred = int(np.argmax(pillar_c_probs))
    agreement = (a_pred == b_pred) and (b_pred == c_pred)

    return pred_idx, confidence, agreement, fused, sim_probs
