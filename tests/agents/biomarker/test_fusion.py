"""Tests for Layer 2 fusion."""

import numpy as np

from immunosense.agents.biomarker.layer2.fusion import fuse_predictions


def test_fusion_returns_valid_distribution():
    class_names = ["A", "B", "C"]
    sims = {"A": 0.8, "B": 0.5, "C": 0.2}
    p_b = np.array([0.6, 0.3, 0.1])
    p_c = np.array([0.7, 0.2, 0.1])
    pred_idx, conf, agree, fused, sim_probs = fuse_predictions(sims, p_b, p_c, class_names)
    assert 0 <= pred_idx < len(class_names)
    assert 0 <= conf <= 1
    assert abs(fused.sum() - 1.0) < 1e-9


def test_fusion_predicts_a_when_all_pillars_agree():
    """If all three pillars favor A, fusion picks A."""
    class_names = ["A", "B", "C"]
    sims = {"A": 0.9, "B": 0.1, "C": 0.1}
    p_b = np.array([0.8, 0.1, 0.1])
    p_c = np.array([0.85, 0.1, 0.05])
    pred_idx, conf, agree, fused, sim_probs = fuse_predictions(sims, p_b, p_c, class_names)
    assert class_names[pred_idx] == "A"
    assert agree is True


def test_fusion_detects_disagreement():
    """When pillars disagree, agreement flag is False."""
    class_names = ["A", "B", "C"]
    sims = {"A": 0.9, "B": 0.0, "C": 0.0}   # Pillar A picks A
    p_b = np.array([0.1, 0.8, 0.1])           # Pillar B picks B
    p_c = np.array([0.1, 0.1, 0.8])           # Pillar C picks C
    pred_idx, conf, agree, fused, sim_probs = fuse_predictions(sims, p_b, p_c, class_names)
    assert agree is False


def test_fusion_sim_probs_sum_to_one():
    class_names = ["A", "B", "C"]
    sims = {"A": 0.7, "B": 0.5, "C": 0.3}
    p_b = np.array([0.4, 0.3, 0.3])
    p_c = np.array([0.5, 0.3, 0.2])
    _, _, _, _, sim_probs = fuse_predictions(sims, p_b, p_c, class_names)
    assert abs(sim_probs.sum() - 1.0) < 1e-9


def test_fusion_custom_weights():
    """Custom weights should propagate (e.g., weight Pillar A heavily)."""
    class_names = ["A", "B"]
    sims = {"A": 1.0, "B": -1.0}              # Strong Pillar A signal for A
    p_b = np.array([0.1, 0.9])                # Pillar B says B
    p_c = np.array([0.1, 0.9])                # Pillar C says B
    # Default weights would pick B; if we crank Pillar A to 1.0, pick A
    pred_idx, _, _, _, _ = fuse_predictions(
        sims, p_b, p_c, class_names,
        weight_a=1.0, weight_b=0.0, weight_c=0.0,
    )
    assert class_names[pred_idx] == "A"


def test_fusion_numerically_stable_with_extreme_similarities():
    """Softmax with very large values should not overflow."""
    class_names = ["A", "B"]
    sims = {"A": 100.0, "B": -100.0}  # Extreme
    p_b = np.array([0.5, 0.5])
    p_c = np.array([0.5, 0.5])
    pred_idx, conf, _, fused, sim_probs = fuse_predictions(sims, p_b, p_c, class_names)
    # No NaN or inf
    assert np.isfinite(fused).all()
    assert np.isfinite(sim_probs).all()
    assert class_names[pred_idx] == "A"


def test_fusion_low_confidence_when_uniform():
    """All pillars uniform => low confidence."""
    class_names = ["A", "B", "C", "D", "E", "F", "G"]
    sims = {c: 0.5 for c in class_names}
    p_b = np.full(7, 1 / 7)
    p_c = np.full(7, 1 / 7)
    _, conf, _, _, _ = fuse_predictions(sims, p_b, p_c, class_names)
    # Confidence should be near 1/7 ≈ 0.143
    assert conf < 0.2
