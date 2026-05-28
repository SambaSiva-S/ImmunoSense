"""End-to-end tests for BiomarkerAgent."""

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.agents.biomarker import BiomarkerAgent
from immunosense.agents.biomarker.constants import LAYER2_EMBEDDING_DIM
from immunosense.agents.biomarker.layer2 import Layer2Bundle


def test_agent_class_attributes():
    assert BiomarkerAgent.agent_id == "agent1_biomarker"
    assert BiomarkerAgent.output_dim == 7
    assert BiomarkerAgent.poll_frequency == "weekly"


def test_agent_init_no_models():
    agent = BiomarkerAgent(patient_id="p001")
    assert agent.patient_id == "p001"
    assert agent.crp_baseline is None
    assert agent.layer2 is None
    # Layer 3 always exists (in-memory state)
    assert agent.layer3 is not None


def test_agent_load_models_layer2_only(trained_layer2_dir):
    agent = BiomarkerAgent()
    agent.load_models(layer2_dir=trained_layer2_dir, require_layer1=False)
    assert agent.layer2 is not None
    assert agent.crp_baseline is None


def test_agent_load_models_missing_layer1_with_require_raises():
    from pathlib import Path
    agent = BiomarkerAgent()
    with pytest.raises(FileNotFoundError):
        agent.load_models(
            layer1_dir=Path("/nonexistent"),
            require_layer1=True,
        )


def test_agent_process_returns_agent_output(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent(patient_id="p001")
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    assert isinstance(output, AgentOutput)
    assert output.agent_id == "agent1_biomarker"


def test_agent_output_vector_is_7_dim(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    assert output.vector.shape == (7,)
    assert output.vector_dim == 7


def test_agent_output_vector_sums_to_one(trained_layer2_dir, ra_reading):
    """Disease probability vector should sum to ~1.0 (it's a distribution)."""
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    assert abs(output.vector.sum() - 1.0) < 1e-3


def test_agent_emit_embedding_returns_128_dim(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    embedding = agent.emit_embedding()
    assert embedding.shape == (128,)
    # Should be on unit sphere
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3


def test_agent_emit_embedding_zero_before_inference():
    """Before any process() call, embedding is a zero vector."""
    agent = BiomarkerAgent()
    embedding = agent.emit_embedding()
    assert embedding.shape == (LAYER2_EMBEDDING_DIM,)
    assert np.allclose(embedding, 0.0)


def test_agent_predicts_ra_for_ra_shaped_reading(trained_layer2_dir, ra_reading):
    """An RA-shaped reading (high CRP/RF/Anti-CCP) should be classified as RA."""
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    assert output.data["layer2"]["prediction"] == "Rheumatoid Arthritis"


def test_agent_predicts_normal_for_normal_reading(trained_layer2_dir, normal_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 30, "sex": 1, "bmi": 23},
        "reading": normal_reading,
    })
    # Should be Normal or at least not RA
    pred = output.data["layer2"]["prediction"]
    assert pred in ["Normal", "Ankylosing Spondylitis", "Sjogren's Syndrome"]


def test_agent_top_drivers_present(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    output = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    drivers = output.data["layer2"]["top_drivers"]
    assert len(drivers) == 3
    for d in drivers:
        assert "feature_name" in d
        assert "shap_value" in d


def test_agent_layer3_personal_data_grows(trained_layer2_dir, ra_reading):
    """Process multiple readings; Layer 3 personal_weight should ramp up."""
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    weights = []
    for i in range(15):
        reading = dict(ra_reading)
        reading["day"] = i
        out = agent.process({
            "demographics": {"age": 52, "sex": 2, "bmi": 26},
            "reading": reading,
        })
        weights.append(out.data["layer3"]["personal_weight"])
    # Weight should grow monotonically (or at least end > start)
    assert weights[-1] > weights[0]
    assert weights[-1] <= 0.8


def test_agent_process_missing_demographics_raises(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    with pytest.raises(ValueError, match="demographics"):
        agent.process({"reading": ra_reading})


def test_agent_process_missing_reading_raises(trained_layer2_dir):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    with pytest.raises(ValueError, match="reading"):
        agent.process({"demographics": {"age": 30, "sex": 1, "bmi": 25}})


def test_agent_get_output_vector_returns_latest(trained_layer2_dir, ra_reading):
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    out1 = agent.process({
        "demographics": {"age": 52, "sex": 2, "bmi": 26},
        "reading": ra_reading,
    })
    cached_vec = agent.get_output_vector()
    assert np.allclose(out1.vector, cached_vec)


def test_agent_runs_30_readings_without_error(trained_layer2_dir, ra_reading):
    """Smoke test: 30 readings should complete without raising."""
    agent = BiomarkerAgent()
    agent.layer2 = Layer2Bundle.load(trained_layer2_dir)
    for i in range(30):
        reading = dict(ra_reading)
        reading["day"] = i
        # Vary CRP slightly to provide signal for Layer 3
        reading["CRP"] = 22.0 + (i % 5 - 2) * 2.0
        out = agent.process({
            "demographics": {"age": 52, "sex": 2, "bmi": 26},
            "reading": reading,
        })
        assert out is not None
