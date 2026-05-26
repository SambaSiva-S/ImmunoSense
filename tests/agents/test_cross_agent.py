"""Cross-agent integration tests.

Validates that Agent 3 (Environment) and Agent 5 (Symptoms & Mood) work together
as the Conductor will use them in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from immunosense.agents.base import BaseAgent
from immunosense.agents.environment import (
    EnvironmentAgent,
    MockEnvironmentSource,
)
from immunosense.agents.environment import (
    process_environment_day as proc_env,
)
from immunosense.agents.symptoms_mood import (
    CompositeSymptomSource,
    MockSymptomSource,
    SymptomsMoodAgent,
)
from immunosense.agents.symptoms_mood import (
    process_symptom_day as proc_sym,
)
from immunosense.agents.wearable import (
    MockWearableGenerator,
    WearableAgent,
)


class TestCrossAgent:
    """Test that agents work together as the Conductor will use them."""

    def test_all_three_agents_inherit_base(self):
        env_agent = EnvironmentAgent()
        sym_agent = SymptomsMoodAgent()
        wea_agent = WearableAgent()
        assert isinstance(env_agent, BaseAgent)
        assert isinstance(sym_agent, BaseAgent)
        assert isinstance(wea_agent, BaseAgent)

    def test_distinct_agent_ids(self):
        assert EnvironmentAgent.agent_id == "agent3_environment"
        assert SymptomsMoodAgent.agent_id == "agent5_symptoms_mood"
        assert WearableAgent.agent_id == "agent4_wearable"

    def test_distinct_output_dims(self):
        assert EnvironmentAgent.output_dim == 5
        assert SymptomsMoodAgent.output_dim == 36
        assert WearableAgent.output_dim == 29

    def test_distinct_embedding_versions(self):
        env = EnvironmentAgent()
        sym = SymptomsMoodAgent()
        wea = WearableAgent()
        versions = {env.embedding_version, sym.embedding_version, wea.embedding_version}
        assert len(versions) == 3  # all distinct

    def test_agent5_flare_distributed_to_agent3(self):
        """Simulate Conductor distributing Agent 5's flare_score to Agent 3."""
        env_agent = EnvironmentAgent()
        sym_agent = SymptomsMoodAgent(patient_id="p001")

        env_mock = MockEnvironmentSource(seed_offset=42)
        sym_mock = MockSymptomSource(disease="RA", seed_offset=42)
        sym_composite = CompositeSymptomSource(mock_fallback=sym_mock)

        start = pd.Timestamp("2026-03-01")

        # 30 days of parallel observation
        for day in range(30):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")

            # Agent 5: produces flare_score
            sym_summary = proc_sym(
                "p001", date, "RA",
                composite_source=sym_composite,
            )
            sym_agent.observe(sym_summary)

            # Conductor takes Agent 5's flare_score and gives it to Agent 3
            env_summary = proc_env("28202", date, source=env_mock)
            env_agent.observe(env_summary)
            env_agent.observe_flare(date, sym_summary.flare_score)

        # Both agents should have 30 days of observation
        assert len(sym_agent.daily_records) == 30
        assert len(env_agent.daily_records) == 30

        # Agent 3's flare_events were populated from Agent 5's scores
        assert len(env_agent.flare_events) == 30

        # Both agents can produce reports
        env_report = env_agent.analyze()
        sym_report = sym_agent.analyze()

        assert env_report.n_days_observed == 30
        assert sym_report.n_days_observed == 30
        assert sym_report.n_hypotheses_tested == 10  # Agent 5 pre-registered

    def test_jepa_embeddings_have_correct_dimensions(self):
        """Verify each agent produces JEPA-compatible vector with declared dim."""
        env_agent = EnvironmentAgent()
        sym_agent = SymptomsMoodAgent()

        env_mock = MockEnvironmentSource(seed_offset=5)
        sym_mock = MockSymptomSource(disease="Mixed")
        sym_composite = CompositeSymptomSource(mock_fallback=sym_mock)

        env_summary = proc_env("28202", "2026-04-01", source=env_mock)
        sym_summary = proc_sym(
            "p001", "2026-04-01", "Mixed",
            composite_source=sym_composite,
        )

        env_agent.observe(env_summary)
        sym_agent.observe(sym_summary)

        env_vec = env_agent.get_output_vector()
        sym_vec = sym_agent.get_output_vector()

        assert env_vec.shape == (5,)
        assert sym_vec.shape == (36,)

        # Both should be valid finite values (or NaN for env cold start which is fine)
        # sym should be all finite
        assert np.isfinite(sym_vec).all()

    def test_both_agents_use_bh_fdr_from_shared_module(self):
        """Sanity check: both agents' detectors use BH FDR (same approach)."""
        # Both should expose tracker_activation with trigger_detector key
        env_agent = EnvironmentAgent()
        sym_agent = SymptomsMoodAgent()

        env_report = env_agent.analyze()
        sym_report = sym_agent.analyze()

        assert "trigger_detector" in env_report.tracker_activation
        assert "trigger_detector" in sym_report.tracker_activation

    def test_agent5_special_outputs_unique(self):
        """Agent 5 exposes flare_score, jepa_embedding, raw_hypothesis_evidence.

        These are Agent 5's special role outputs that other agents don't have.
        """
        sym_agent = SymptomsMoodAgent()

        # daily_flare_score
        assert sym_agent.daily_flare_score() == 0.0
        # jepa_embedding
        emb = sym_agent.jepa_embedding()
        assert emb.shape == (36,)
        # raw_hypothesis_evidence
        assert sym_agent.raw_hypothesis_evidence() == []

        # Agent 3 should NOT have these methods
        env_agent = EnvironmentAgent()
        assert not hasattr(env_agent, "daily_flare_score")
        assert not hasattr(env_agent, "raw_hypothesis_evidence")
