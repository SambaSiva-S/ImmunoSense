"""Tests for the JEPA embedding envelope (Challenge 5).

Stable concatenation layout, dim validation, presence mask, and the
build_patient_day_embedding assembly path.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.inference import (
    EMBEDDING_LAYOUT_VERSION,
    JEPACompatible,
    PatientDayEmbedding,
    TOTAL_CONCAT_DIM,
    build_patient_day_embedding,
)


def _out(agent_id, dim, scale=1.0):
    return AgentOutput(
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        data={},
        vector=np.ones(dim) * scale,
        vector_dim=dim,
    )


class TestLayoutContract:
    def test_total_concat_dim(self):
        # biomarker(7) + dietary(10) + environment(5) + wearable(29) + symptoms(36) = 87
        assert TOTAL_CONCAT_DIM == 87

    def test_layout_version_recorded(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        assert pde.layout_version == EMBEDDING_LAYOUT_VERSION


class TestAddAndConcat:
    def test_empty_concat_is_zero_vector(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        v = pde.to_concat()
        assert v.shape == (TOTAL_CONCAT_DIM,)
        assert np.allclose(v, 0.0)
        assert pde.n_present == 0

    def test_single_agent_block_position(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        # biomarker is the FIRST slot (dim 7).
        pde.add("agent1_biomarker", np.ones(7) * 0.5)
        v = pde.to_concat()
        # First 7 values are the biomarker block.
        assert np.allclose(v[:7], 0.5)
        # Rest is zero.
        assert np.allclose(v[7:], 0.0)

    def test_dim_mismatch_rejected_on_add(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        with pytest.raises(ValueError):
            pde.add("agent1_biomarker", np.ones(5))  # wrong dim (7 expected)

    def test_presence_mask(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        pde.add("agent1_biomarker", np.ones(7))
        pde.add("agent5_symptoms_mood", np.ones(36))
        mask = pde.presence_mask()
        # Order: biomarker, dietary, environment, wearable, symptoms_mood
        assert mask.tolist() == [True, False, False, False, True]

    def test_full_concat_when_all_present(self):
        pde = PatientDayEmbedding(user_id="p", bucket_id="b")
        pde.add("agent1_biomarker", np.full(7, 0.1))
        pde.add("agent2_dietary", np.full(10, 0.2))
        pde.add("agent3_environment", np.full(5, 0.3))
        pde.add("agent4_wearable", np.full(29, 0.4))
        pde.add("agent5_symptoms_mood", np.full(36, 0.5))
        v = pde.to_concat()
        assert v.shape == (TOTAL_CONCAT_DIM,)
        assert np.allclose(v[:7], 0.1)
        assert np.allclose(v[7:17], 0.2)
        assert np.allclose(v[17:22], 0.3)
        assert np.allclose(v[22:51], 0.4)
        assert np.allclose(v[51:87], 0.5)
        assert pde.n_present == 5


class TestBuildFromOutputs:
    def test_assembles_from_outputs(self):
        outputs = {
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, 0.5),
            "agent1_biomarker": _out("agent1_biomarker", 7, 0.3),
        }
        pde = build_patient_day_embedding("p", "b", outputs)
        assert pde.n_present == 2
        v = pde.to_concat()
        assert np.allclose(v[:7], 0.3)        # biomarker
        assert np.allclose(v[7:22], 0.0)      # absent dietary+env
        assert np.allclose(v[22:51], 0.0)     # absent wearable
        assert np.allclose(v[51:87], 0.5)     # symptoms

    def test_unknown_agent_skipped_no_corruption(self):
        outputs = {
            "agent99_mystery": _out("agent99_mystery", 5),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, 0.5),
        }
        pde = build_patient_day_embedding("p", "b", outputs)
        assert pde.n_present == 1
        # Concat is still well-formed.
        assert pde.to_concat().shape == (TOTAL_CONCAT_DIM,)

    def test_dim_mismatch_skipped_silently(self):
        # Assembly path is forgiving; explicit add() is strict.
        wrong = _out("agent1_biomarker", 5)  # should be 7
        pde = build_patient_day_embedding("p", "b", {"agent1_biomarker": wrong})
        # Mismatched agent's slot stays zero, no exception.
        assert pde.n_present == 0
        assert np.allclose(pde.to_concat(), 0.0)

    def test_empty_outputs(self):
        pde = build_patient_day_embedding("p", "b", {})
        assert pde.n_present == 0
        assert pde.to_concat().shape == (TOTAL_CONCAT_DIM,)


class TestJEPACompatibleProtocol:
    def test_real_agents_satisfy_protocol(self):
        # The real BaseAgent subclasses implement emit_embedding, so they
        # should structurally satisfy JEPACompatible.
        from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent
        agent = SymptomsMoodAgent()
        assert isinstance(agent, JEPACompatible)
