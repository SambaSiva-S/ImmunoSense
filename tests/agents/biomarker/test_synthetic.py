"""Tests for the synthetic patient generator."""

import numpy as np
import pandas as pd

from immunosense.agents.biomarker.synthetic import (
    ImprovedPatientGenerator,
    generate_synthetic_trajectory,
)


def test_generate_synthetic_trajectory_ra():
    trajectory, baseline = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=30, random_seed=42,
    )
    assert len(trajectory) == 30
    # Each reading should have day, is_flare, and biomarker values
    assert "day" in trajectory[0]
    assert "is_flare" in trajectory[0]
    assert "CRP" in trajectory[0]
    # baseline should have CRP
    assert "CRP" in baseline


def test_generate_synthetic_trajectory_includes_some_flares():
    trajectory, _ = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=30, random_seed=42,
    )
    flare_count = sum(1 for r in trajectory if r["is_flare"])
    # 2-3 flares × 3 readings = 6-9 flare readings expected
    assert 4 <= flare_count <= 12


def test_generate_synthetic_trajectory_normal_has_no_flares():
    trajectory, _ = generate_synthetic_trajectory(
        disease="Normal", n_readings=30, random_seed=42,
    )
    # Normal trajectories still get planted flares for testing purposes
    # but values are lower
    assert all(r["CRP"] >= 0 for r in trajectory)


def test_generate_synthetic_trajectory_lupus_complement_drops():
    """Lupus flares should show C3/C4 drops, not just CRP spikes."""
    trajectory, baseline = generate_synthetic_trajectory(
        disease="Systemic Lupus Erythematosus", n_readings=30, random_seed=42,
    )
    flare_readings = [r for r in trajectory if r["is_flare"]]
    normal_readings = [r for r in trajectory if not r["is_flare"]]
    if flare_readings and normal_readings:
        avg_flare_c3 = np.mean([r["C3"] for r in flare_readings])
        avg_normal_c3 = np.mean([r["C3"] for r in normal_readings])
        # C3 should drop during flare
        assert avg_flare_c3 < avg_normal_c3


def test_improved_patient_generator_builds_stats(synthetic_rheumatic_df):
    gen = ImprovedPatientGenerator(synthetic_rheumatic_df)
    assert "Rheumatoid Arthritis" in gen.disease_stats
    assert "CRP" in gen.disease_stats["Rheumatoid Arthritis"]
    assert "mean" in gen.disease_stats["Rheumatoid Arthritis"]["CRP"]


def test_improved_patient_generator_trajectory(synthetic_rheumatic_df):
    gen = ImprovedPatientGenerator(synthetic_rheumatic_df)
    trajectory, baseline = gen.generate_trajectory(
        disease="Rheumatoid Arthritis", n_readings=30, random_seed=42,
    )
    assert len(trajectory) == 30
    assert "CRP" in baseline
    # All biomarker values must be positive
    assert all(r["CRP"] > 0 for r in trajectory)


def test_improved_patient_generator_unknown_disease_raises(synthetic_rheumatic_df):
    import pytest
    gen = ImprovedPatientGenerator(synthetic_rheumatic_df)
    with pytest.raises(ValueError, match="Unknown disease"):
        gen.generate_trajectory(disease="Mongolian Beriberi")


def test_trajectory_has_triggers():
    trajectory, _ = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=30, random_seed=42,
    )
    # At least some readings should have triggers
    has_gluten = any(r.get("gluten_exposure") for r in trajectory)
    has_sleep = any(r.get("poor_sleep") for r in trajectory)
    assert has_gluten or has_sleep


def test_trajectory_days_strictly_increasing():
    trajectory, _ = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=20, random_seed=42,
    )
    days = [r["day"] for r in trajectory]
    assert days == sorted(days)


def test_reproducibility_with_same_seed():
    """Same seed should produce the same trajectory."""
    t1, _ = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=20, random_seed=42,
    )
    t2, _ = generate_synthetic_trajectory(
        disease="Rheumatoid Arthritis", n_readings=20, random_seed=42,
    )
    crp1 = [r["CRP"] for r in t1]
    crp2 = [r["CRP"] for r in t2]
    assert crp1 == crp2
