"""Tests for corroboration patterns (Challenge 3 Phase 2).

Critical invariant: corroboration is SEMANTIC ONLY. It must never expose a
probability or any signal that could feed back into the Bayesian fusion.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from immunosense.agents.base import AgentOutput
from immunosense.conductor.fusion.corroboration import (
    Corroboration,
    CorroborationPattern,
    MatchedPattern,
)


def _out(agent_id, dim, alerts=None):
    return AgentOutput(
        agent_id=agent_id,
        timestamp=datetime.now(timezone.utc),
        data={},
        vector=np.ones(dim),
        vector_dim=dim,
        alerts=alerts or [],
        confidence=0.9,
    )


_CRIT = [{"severity": "critical"}]


class TestPatternLibrary:
    def setup_method(self):
        self.corr = Corroboration()

    def test_library_size_v1(self):
        # Q2 decision: 6-8 cross-disease patterns in v1.
        assert 6 <= len(self.corr.patterns) <= 8

    def test_patterns_have_provenance(self):
        for p in self.corr.patterns:
            assert p.source, f"{p.name} missing source"
            assert p.required_agents, f"{p.name} has no required_agents"
            assert p.description, f"{p.name} missing description"

    def test_patterns_have_stable_names(self):
        names = [p.name for p in self.corr.patterns]
        assert len(names) == len(set(names)), "pattern names must be unique"
        for n in names:
            assert " " not in n and n == n.lower()


class TestMatching:
    def setup_method(self):
        self.corr = Corroboration()

    def test_autonomic_stress_matches(self):
        outputs = {
            "agent4_wearable": _out("agent4_wearable", 29, _CRIT),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT),
        }
        matched = self.corr.match(outputs)
        names = [m.name for m in matched]
        assert "autonomic_stress" in names

    def test_single_agent_no_match(self):
        # autonomic_stress needs BOTH wearable + symptoms; only one shouldn't match.
        outputs = {"agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT)}
        matched = self.corr.match(outputs)
        assert all(m.name != "autonomic_stress" for m in matched)

    def test_no_elevation_no_match(self):
        # Both agents present but no alerts -> low signal -> no match.
        outputs = {
            "agent4_wearable": _out("agent4_wearable", 29, alerts=[]),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, alerts=[]),
        }
        # No vector signal either (vector=ones is signal ~0.6 — actually elevated).
        # So test with explicitly low vector.
        for o in outputs.values():
            o.vector = np.zeros(o.vector_dim)
        matched = self.corr.match(outputs)
        # The zero-vector outputs have signal 0, well below participation threshold.
        assert matched == []

    def test_multi_system_pattern(self):
        outputs = {
            "agent1_biomarker": _out("agent1_biomarker", 7, _CRIT),
            "agent4_wearable": _out("agent4_wearable", 29, _CRIT),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT),
        }
        matched = self.corr.match(outputs)
        names = [m.name for m in matched]
        assert "physiological_multisystem" in names
        # Should also catch other 2-agent patterns within these three.
        assert "autonomic_stress" in names
        assert "inflammatory_surge" in names

    def test_optional_agents_recorded_when_elevated(self):
        # inflammatory_surge requires biomarker+symptoms, optionally wearable.
        outputs = {
            "agent1_biomarker": _out("agent1_biomarker", 7, _CRIT),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT),
            "agent4_wearable": _out("agent4_wearable", 29, _CRIT),
        }
        matched = self.corr.match(outputs)
        infl = [m for m in matched if m.name == "inflammatory_surge"][0]
        assert "agent4_wearable" in infl.participating_agents

    def test_silent_physiological_no_symptoms(self):
        # silent_physiological needs biomarker+wearable; works WITHOUT symptoms.
        outputs = {
            "agent1_biomarker": _out("agent1_biomarker", 7, _CRIT),
            "agent4_wearable": _out("agent4_wearable", 29, _CRIT),
        }
        matched = self.corr.match(outputs)
        assert "silent_physiological" in [m.name for m in matched]


class TestNoMathFeedback:
    """Phase 2 must NEVER expose a probability or feed the Bayesian math."""

    def test_matched_pattern_has_no_probability_field(self):
        corr = Corroboration()
        outputs = {
            "agent4_wearable": _out("agent4_wearable", 29, _CRIT),
            "agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT),
        }
        matched = corr.match(outputs)
        for m in matched:
            # Defensive: ensure no probability-like field leaks in.
            for forbidden in ("probability", "flare_probability", "lr", "log_lr",
                              "likelihood_ratio", "weight", "score"):
                assert not hasattr(m, forbidden), (
                    f"corroboration MatchedPattern must not expose {forbidden!r} "
                    f"(would risk double-counting with Phase 1)"
                )

    def test_match_returns_only_matched_patterns(self):
        corr = Corroboration()
        matched = corr.match({})
        assert isinstance(matched, list)
        assert all(isinstance(m, MatchedPattern) for m in matched)


class TestCustomPatterns:
    def test_can_inject_custom_library(self):
        custom = [
            CorroborationPattern(
                name="custom_test",
                label="Custom",
                required_agents=("agent5_symptoms_mood",),
                description="test",
                source="test",
            )
        ]
        corr = Corroboration(patterns=custom)
        outputs = {"agent5_symptoms_mood": _out("agent5_symptoms_mood", 36, _CRIT)}
        matched = corr.match(outputs)
        assert len(matched) == 1
        assert matched[0].name == "custom_test"
