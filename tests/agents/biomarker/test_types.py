"""Tests for biomarker dataclass types."""

from immunosense.agents.biomarker.types import (
    BiomarkerAgentReport,
    BiomarkerReading,
    DetectedTriggerPattern,
    Layer1Output,
    Layer2Output,
    Layer3Output,
)


def test_biomarker_reading_defaults_to_none():
    r = BiomarkerReading()
    assert r.day == 0
    assert r.CRP is None
    assert r.is_flare is False
    assert r.extra == {}


def test_biomarker_reading_accepts_values():
    r = BiomarkerReading(day=14, CRP=12.0, ESR=45, is_flare=True)
    assert r.day == 14
    assert r.CRP == 12.0
    assert r.is_flare is True


def test_layer1_output_fields():
    out = Layer1Output(
        biomarker="CRP", value=12.0,
        population_percentile=0.95, interpretation="ALARMING",
    )
    assert out.biomarker == "CRP"
    assert out.population_percentile == 0.95
    assert out.interpretation == "ALARMING"


def test_layer2_output_has_all_pillars():
    out = Layer2Output(
        prediction="Rheumatoid Arthritis",
        confidence=0.85,
        probabilities={"Rheumatoid Arthritis": 0.85, "Normal": 0.15},
        pillar_a_similarities={"Rheumatoid Arthritis": 0.9, "Normal": 0.1},
        pillar_b_probabilities={"Rheumatoid Arthritis": 0.8, "Normal": 0.2},
        pillar_c_probabilities={"Rheumatoid Arthritis": 0.9, "Normal": 0.1},
        pillars_agree=True,
        contrastive_embedding=[0.1] * 128,
        top_drivers=[],
    )
    assert out.prediction == "Rheumatoid Arthritis"
    assert out.pillars_agree is True
    assert len(out.contrastive_embedding) == 128


def test_layer2_output_top_drivers_default_empty():
    out = Layer2Output(
        prediction="Normal", confidence=0.9,
        probabilities={}, pillar_a_similarities={},
        pillar_b_probabilities={}, pillar_c_probabilities={},
        pillars_agree=True,
    )
    assert out.top_drivers == []


def test_detected_trigger_pattern_fields():
    p = DetectedTriggerPattern(
        trigger="poor_sleep", biomarker="CRP", lag_readings=2,
        correlation=0.65, effect_size=4.5, effect_pct=120.0,
        n_exposed=8, strength="STRONG",
    )
    assert p.trigger == "poor_sleep"
    assert p.strength == "STRONG"


def test_layer3_output_no_personal_data_initially():
    out = Layer3Output(
        has_personal_data=False, readings_count=0, personal_weight=0.0,
        biomarkers={},
    )
    assert out.has_personal_data is False
    assert out.personal_weight == 0.0
    assert out.patterns == []
    assert out.flare_rule is None


def test_biomarker_agent_report_fields():
    report = BiomarkerAgentReport(
        timestamp=14, layer1={}, layer2=None,
        layer3=Layer3Output(
            has_personal_data=False, readings_count=0,
            personal_weight=0.0, biomarkers={},
        ),
        alerts=[],
    )
    assert report.timestamp == 14
    assert report.layer2 is None
    assert report.alerts == []
