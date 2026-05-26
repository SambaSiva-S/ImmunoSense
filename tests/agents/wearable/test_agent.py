"""End-to-end tests for WearableAgent.

Includes the 30-night integration test from the original notebook to verify
the extracted package reproduces the original behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from immunosense.agents.base import BaseAgent
from immunosense.agents.wearable import (
    MockWearableGenerator,
    WearableAgent,
    derive_physiological_states,
)


class TestWearableAgentBasics:
    def test_inherits_baseagent(self):
        agent = WearableAgent()
        assert isinstance(agent, BaseAgent)

    def test_agent_identity(self):
        agent = WearableAgent()
        assert agent.agent_id == "agent4_wearable"
        assert agent.output_dim == 29
        assert agent.embedding_version == "agent4_wearable_v1.0.0"

    def test_cold_start_vector_is_nan(self):
        agent = WearableAgent()
        v = agent.get_output_vector()
        assert v.shape == (29,)
        assert np.isnan(v).all()


class TestMockWearableGenerator:
    def test_generates_480_minutes(self):
        gen = MockWearableGenerator(seed=42)
        night_df, rr = gen.generate_night(0, flare_state="normal")
        assert len(night_df) == 480
        assert len(rr) > 0

    def test_required_columns(self):
        gen = MockWearableGenerator(seed=42)
        night_df, _ = gen.generate_night(0)
        for col in ["timestamp", "hr", "skin_temp", "enmo", "spo2", "sleep_stage"]:
            assert col in night_df.columns

    def test_prodrome_has_higher_hr(self):
        gen1 = MockWearableGenerator(seed=42)
        normal_df, _ = gen1.generate_night(0, flare_state="normal")
        gen2 = MockWearableGenerator(seed=42)
        prodrome_df, _ = gen2.generate_night(0, flare_state="prodrome")
        # Prodrome has +4 hr_offset
        assert prodrome_df["hr"].mean() > normal_df["hr"].mean()

    def test_deterministic_with_same_seed(self):
        gen1 = MockWearableGenerator(seed=42)
        df1, rr1 = gen1.generate_night(0, flare_state="normal")
        gen2 = MockWearableGenerator(seed=42)
        df2, rr2 = gen2.generate_night(0, flare_state="normal")
        # Same seed should give same data
        assert (df1["hr"].values == df2["hr"].values).all()
        assert rr1[:10] == rr2[:10]


class TestWearableAgentProcess:
    def test_single_night_produces_valid_output(self):
        agent = WearableAgent()
        gen = MockWearableGenerator(seed=42)
        night_df, rr = gen.generate_night(0, flare_state="normal")
        out = agent.process({
            "night_df": night_df,
            "rr_intervals": rr,
            "night_idx": 0,
        })
        assert out.agent_id == "agent4_wearable"
        assert out.vector.shape == (29,)
        assert out.vector_dim == 29
        assert 0.0 <= out.confidence <= 1.0
        assert out.trace_id.startswith("agent4_wearable-")
        assert "physiological_states" in out.data

    def test_missing_input_raises(self):
        agent = WearableAgent()
        with pytest.raises(KeyError):
            agent.process({})

    def test_latency_tracked(self):
        agent = WearableAgent()
        gen = MockWearableGenerator(seed=42)
        # Process 3 nights
        for i in range(3):
            night_df, rr = gen.generate_night(i)
            agent.process({"night_df": night_df, "rr_intervals": rr, "night_idx": i})
        status = agent.get_status()
        assert status.avg_latency_ms > 0
        assert status.status == "healthy"


class TestEndToEnd30Nights:
    """Reproduce the integration test from the original notebook."""

    @pytest.fixture
    def trajectory(self):
        agent = WearableAgent()
        gen = MockWearableGenerator(patient_id="p001", seed=42)
        flare_schedule = ["normal"] * 30
        flare_schedule[10] = "acute_stress"
        flare_schedule[20] = "prodrome"
        flare_schedule[21] = "prodrome"

        outputs = []
        for night_idx, state in enumerate(flare_schedule):
            night_df, rr = gen.generate_night(night_idx, flare_state=state)
            out = agent.process({
                "night_df": night_df,
                "rr_intervals": rr,
                "night_idx": night_idx,
                "is_flare": state != "normal",
            })
            outputs.append((state, out))
        return outputs

    def test_all_nights_produced_output(self, trajectory):
        assert len(trajectory) == 30

    def test_at_least_one_prodrome_alert(self, trajectory):
        n_prodrome = sum(
            1 for s, o in trajectory
            if any(a["name"] == "autoimmune_prodrome" for a in o.alerts)
        )
        assert n_prodrome >= 1

    def test_at_least_one_acute_stress_alert(self, trajectory):
        n_acute = sum(
            1 for s, o in trajectory
            if any(a["name"] == "acute_stress_response" for a in o.alerts)
        )
        assert n_acute >= 1

    def test_normal_nights_have_few_alerts(self, trajectory):
        # Normal nights should rarely have critical alerts
        normal_critical = sum(
            1 for s, o in trajectory
            if s == "normal"
            and any(a["severity"] == "critical" for a in o.alerts)
        )
        # Allow some noise but not many
        assert normal_critical <= 2

    def test_all_outputs_have_29_dim_vector(self, trajectory):
        for state, out in trajectory:
            assert out.vector.shape == (29,)


class TestPhysiologicalStates:
    def test_low_hrv_yields_vagal_withdrawal(self):
        reading = {"hrv_sleep_vs_baseline_ratio": 0.4}
        states = derive_physiological_states(reading, [])
        assert states["autonomic_state"] == "severe_vagal_withdrawal"

    def test_balanced_when_hrv_normal(self):
        reading = {"hrv_sleep_vs_baseline_ratio": 1.0}
        states = derive_physiological_states(reading, [])
        assert states["autonomic_state"] == "balanced"

    def test_critical_alert_sets_overall_alert(self):
        reading = {}
        alerts = [{"severity": "critical"}]
        states = derive_physiological_states(reading, alerts)
        assert states["overall_state"] == "alert"

    def test_warning_alert_sets_overall_concerning(self):
        reading = {}
        alerts = [{"severity": "warning"}]
        states = derive_physiological_states(reading, alerts)
        assert states["overall_state"] == "concerning"

    def test_high_temp_signals_inflammation(self):
        reading = {"skin_temp_deviation": 0.8}
        states = derive_physiological_states(reading, [])
        assert states["inflammation_signal"] == "active"

    def test_quiet_when_temp_normal(self):
        reading = {"skin_temp_deviation": 0.05}
        states = derive_physiological_states(reading, [])
        assert states["inflammation_signal"] == "quiet"


class TestPatternDetector:
    def test_too_few_readings(self):
        from immunosense.agents.wearable import PatternDetector
        pd_obj = PatternDetector(
            features=["hrv_rmssd_sleep"],
            triggers=["poor_sleep"],
        )
        result = pd_obj.analyze([{"hrv_rmssd_sleep": 30.0}])
        assert not result["has_patterns"]
