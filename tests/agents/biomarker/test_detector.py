"""Tests for Layer 3 PatternDetector."""

import numpy as np

from immunosense.agents.biomarker.layer3.detector import PatternDetector


def make_correlated_trajectory(n=30, lag=1, correlation_strength=1.0, seed=0):
    """Build a trajectory where 'trigger' at t -> 'CRP' spike at t+lag."""
    rng = np.random.RandomState(seed)
    trajectory = []
    triggers = rng.rand(n) < 0.3
    for i in range(n):
        # CRP is baseline 2 + spike if trigger fired `lag` readings ago
        crp = 2.0 + rng.normal(0, 0.2)
        if i - lag >= 0 and triggers[i - lag]:
            crp += 10.0 * correlation_strength
        trajectory.append({
            "day": i,
            "CRP": crp,
            "ESR": 10 + rng.normal(0, 1.5),
            "RF": 8 + rng.normal(0, 0.5),
            "Anti-CCP": 3 + rng.normal(0, 0.3),
            "C3": 120 + rng.normal(0, 3),
            "C4": 28 + rng.normal(0, 1),
            "gluten_exposure": bool(triggers[i]),
            "is_flare": bool((i - lag >= 0 and triggers[i - lag])),
        })
    return trajectory


def test_no_patterns_when_not_enough_readings():
    detector = PatternDetector(min_readings=10)
    result = detector.analyze([{"CRP": 2.0}] * 5)
    assert result["has_patterns"] is False


def test_detects_strong_correlation():
    """A planted strong correlation should be picked up."""
    trajectory = make_correlated_trajectory(n=30, lag=1, correlation_strength=1.0)
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    assert result["has_patterns"] is True
    # The top pattern should be gluten_exposure -> CRP at lag 1
    top = result["patterns"][0]
    assert top["trigger"] == "gluten_exposure"
    assert top["biomarker"] == "CRP"


def test_correlation_strength_strong_when_above_threshold():
    """Above |r|=0.5 should be marked STRONG."""
    trajectory = make_correlated_trajectory(n=30, lag=1, correlation_strength=1.0)
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    strong_patterns = [p for p in result["patterns"] if p["strength"] == "STRONG"]
    assert len(strong_patterns) >= 1


def test_patterns_sorted_by_correlation_descending():
    trajectory = make_correlated_trajectory(n=30, correlation_strength=1.0)
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    patterns = result["patterns"]
    if len(patterns) > 1:
        abs_correlations = [abs(p["correlation"]) for p in patterns]
        assert abs_correlations == sorted(abs_correlations, reverse=True)


def test_flare_rule_derived_when_flare_present():
    trajectory = make_correlated_trajectory(n=30, correlation_strength=1.0)
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    flare_rule = result["flare_rule"]
    # We have flares planted in the synthetic trajectory
    assert flare_rule is not None
    # CRP should have a higher flare_mean than normal_mean
    if "CRP" in flare_rule:
        assert flare_rule["CRP"]["flare_mean"] > flare_rule["CRP"]["normal_mean"]


def test_flare_rule_none_when_no_flares():
    """If no readings are flares, flare_rule should be None."""
    trajectory = [{
        "day": i, "CRP": 2.0, "ESR": 10, "RF": 8,
        "Anti-CCP": 3, "C3": 120, "C4": 28,
        "gluten_exposure": False, "is_flare": False,
    } for i in range(20)]
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    assert result["flare_rule"] is None


def test_returns_n_readings_analyzed():
    trajectory = [{"day": i, "CRP": 2.0, "is_flare": False} for i in range(15)]
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    assert result["n_readings_analyzed"] == 15


def test_no_correlation_when_random():
    """Random uncorrelated data should produce few/no patterns."""
    rng = np.random.RandomState(0)
    trajectory = []
    for i in range(40):
        trajectory.append({
            "day": i,
            "CRP": 2.0 + rng.normal(0, 0.3),
            "ESR": 10 + rng.normal(0, 1.0),
            "RF": 8 + rng.normal(0, 0.4),
            "Anti-CCP": 3 + rng.normal(0, 0.2),
            "C3": 120 + rng.normal(0, 2),
            "C4": 28 + rng.normal(0, 1),
            "gluten_exposure": bool(rng.rand() < 0.3),
            "is_flare": False,
        })
    detector = PatternDetector()
    result = detector.analyze(trajectory)
    # Random data may produce some spurious correlations, but no STRONG ones
    strong = [p for p in result["patterns"] if p["strength"] == "STRONG"]
    assert len(strong) == 0
