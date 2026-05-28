"""Tests for Layer 3 PersonalAdaptationEngine."""

import numpy as np

from immunosense.agents.biomarker.layer3.engine import PersonalAdaptationEngine


def test_engine_starts_with_empty_state():
    engine = PersonalAdaptationEngine()
    assert engine.trajectory_history == []


def test_engine_process_reading_returns_context():
    engine = PersonalAdaptationEngine()
    result = engine.process_reading({"CRP": 2.0, "ESR": 10})
    assert "has_personal_data" in result
    assert "alerts" in result
    assert "patterns" in result


def test_engine_no_personal_data_until_3_readings():
    engine = PersonalAdaptationEngine()
    for _ in range(2):
        result = engine.process_reading({"CRP": 2.0})
        assert result["has_personal_data"] is False


def test_engine_personal_data_active_after_3_readings():
    engine = PersonalAdaptationEngine()
    for _ in range(3):
        result = engine.process_reading({"CRP": 2.0, "ESR": 10})
    assert result["has_personal_data"] is True


def test_engine_no_alerts_when_no_personal_data():
    engine = PersonalAdaptationEngine()
    result = engine.process_reading({"CRP": 2.0})
    assert result["alerts"] == []


def test_engine_critical_alert_on_huge_spike():
    """A severe spike after baseline established should trigger a CRITICAL alert."""
    import random
    random.seed(0)
    engine = PersonalAdaptationEngine()
    # Establish baseline with variance
    for _ in range(10):
        engine.process_reading({
            "CRP": 2.0 + random.uniform(-0.3, 0.3),
            "ESR": 10 + random.uniform(-1.5, 1.5),
            "RF": 8, "Anti-CCP": 3, "C3": 120, "C4": 28,
        })
    # Massive spike
    result = engine.process_reading({"CRP": 50.0, "ESR": 10, "RF": 8, "Anti-CCP": 3, "C3": 120, "C4": 28})
    critical_alerts = [a for a in result["alerts"] if a["level"] == "CRITICAL"]
    assert len(critical_alerts) >= 1


def test_engine_pattern_detection_kicks_in_at_10_readings():
    """Pattern detector only runs after collecting enough data."""
    engine = PersonalAdaptationEngine()
    # First 9 readings -> no patterns yet
    for i in range(9):
        result = engine.process_reading({
            "CRP": 2.0, "ESR": 10, "RF": 8,
            "Anti-CCP": 3, "C3": 120, "C4": 28,
            "gluten_exposure": False,
        })
    assert result["patterns"].get("message") == "Collecting data..." or result["patterns"]["has_patterns"] is False

    # 10th reading -> pattern detector runs
    for i in range(5):
        result = engine.process_reading({
            "CRP": 2.0, "ESR": 10, "RF": 8,
            "Anti-CCP": 3, "C3": 120, "C4": 28,
            "gluten_exposure": False,
        })
    # patterns key should now have a "patterns" list (possibly empty)
    assert "patterns" in result["patterns"]


def test_engine_custom_biomarkers_and_triggers():
    """Engine should accept custom biomarker and trigger lists."""
    engine = PersonalAdaptationEngine(
        biomarkers=["CRP", "ESR"],
        triggers=["poor_sleep"],
    )
    for _ in range(5):
        engine.process_reading({"CRP": 2.0, "ESR": 10, "poor_sleep": False})
    assert engine.tracker.biomarkers == ["CRP", "ESR"]
    assert engine.detector.triggers == ["poor_sleep"]


def test_engine_trajectory_history_grows():
    engine = PersonalAdaptationEngine()
    for _ in range(5):
        engine.process_reading({"CRP": 2.0})
    assert len(engine.trajectory_history) == 5
