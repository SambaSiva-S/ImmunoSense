"""Tests for the three Phase 1 builders.

symptom + biomarker: pure mapping tests (no external data).
dietary: uses tiny synthetic NHANES cache fixtures (server/tests/fixtures/)
so the tests don't depend on the multi-MB real NHANES files.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.builders import (
    build_biomarker_input,
    build_symptom_summary,
    DietaryPipeline,
)

FIXTURES = Path(__file__).parent / "fixtures"
DENSITY = FIXTURES / "density_small.pkl"
FOOD_INDEX = FIXTURES / "food_index_small.pkl"


# --------------------------------------------------------------------------- #
# Symptom builder
# --------------------------------------------------------------------------- #
def _slog(**kw):
    base = dict(
        fatigue=None, joint_pain=None, brain_fog_severity=None, gi_distress=None,
        skin_severity=None, sleep_severity=None, energy_severity=None,
        wellness_severity=None, phq8_score=None, gad7_score=None,
        free_text=None, source="tap",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestSymptomBuilder:
    def test_single_log_direct_mapping(self):
        s = build_symptom_summary([_slog(fatigue=6.0, joint_pain=4.0)],
                                  date="2026-05-27", user_id="u1", disease="SLE")
        assert s.fatigue == 6.0
        assert s.joint_pain == 4.0
        assert s.date == "2026-05-27"
        assert s.disease == "SLE"
        assert s.patient_id == "u1"  # agent-internal field carries user_id

    def test_multi_log_takes_max_per_field(self):
        logs = [_slog(fatigue=5.0, joint_pain=4.0),
                _slog(fatigue=7.0, gi_distress=3.0)]
        s = build_symptom_summary(logs, "2026-05-27", "u1", "SLE")
        assert s.fatigue == 7.0       # max(5,7)
        assert s.joint_pain == 4.0    # only in log 1
        assert s.gi_distress == 3.0   # only in log 2

    def test_free_text_collected(self):
        logs = [_slog(free_text="rough day"), _slog(fatigue=5.0)]
        s = build_symptom_summary(logs, "2026-05-27", "u1", "SLE")
        assert "rough day" in s.new_symptom_mentions

    def test_sources_tracked(self):
        logs = [_slog(source="tap"), _slog(source="voice"), _slog(source="tap")]
        s = build_symptom_summary(logs, "2026-05-27", "u1", "SLE")
        assert s.sources == {"tap": 2, "voice": 1}

    def test_empty_logs(self):
        s = build_symptom_summary([], "2026-05-27", "u1", "SLE")
        assert s.fatigue is None
        assert s.new_symptom_mentions == []

    def test_flare_button_propagated(self):
        s = build_symptom_summary([_slog(fatigue=8.0)], "2026-05-27", "u1", "SLE",
                                  flare_button_pressed=True, explicit_flare_severity=0.9)
        assert s.flare_button_pressed is True
        assert s.flare_score == 0.9

    def test_feeds_real_agent(self):
        from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent
        from immunosense.adapters import SymptomsMoodAdapter
        from immunosense.events.bucket import AgentData
        s = build_symptom_summary([_slog(fatigue=6.0, joint_pain=5.0)],
                                  "2026-05-27", "u1", "SLE")
        res = SymptomsMoodAdapter(SymptomsMoodAgent()).run(
            AgentData("agent5_symptoms_mood", s), bucket_end=datetime.now(timezone.utc))
        assert res.ok
        assert res.output.vector_dim == 36


# --------------------------------------------------------------------------- #
# Biomarker builder
# --------------------------------------------------------------------------- #
class TestBiomarkerBuilder:
    def test_assembles_demographics_and_reading(self):
        reading = SimpleNamespace(crp=8.5, esr=30.0, payload={})
        profile = SimpleNamespace(age=45, sex=2, bmi=27.0)
        out = build_biomarker_input(reading, profile)
        assert out["demographics"] == {"age": 45.0, "sex": 2, "bmi": 27.0}
        assert out["reading"]["CRP"] == 8.5
        assert out["reading"]["ESR"] == 30.0

    def test_payload_merged_into_reading(self):
        reading = SimpleNamespace(crp=5.0, esr=None, payload={"WBC": 7.2, "Hgb": 13.5})
        profile = SimpleNamespace(age=50, sex=1, bmi=24.0)
        out = build_biomarker_input(reading, profile)
        assert out["reading"]["CRP"] == 5.0
        assert "ESR" not in out["reading"]  # was None
        assert out["reading"]["WBC"] == 7.2
        assert out["reading"]["Hgb"] == 13.5

    def test_missing_demographics_defaults(self):
        reading = SimpleNamespace(crp=3.0, esr=None, payload={})
        profile = SimpleNamespace(age=None, sex=None, bmi=None)
        out = build_biomarker_input(reading, profile)
        assert out["demographics"]["age"] == 40.0   # default
        assert out["demographics"]["sex"] == 1       # default
        assert out["demographics"]["bmi"] == 25.0    # default

    def test_feeds_real_agent(self):
        from immunosense.agents.biomarker.agent import BiomarkerAgent
        from immunosense.adapters import BiomarkerAdapter
        from immunosense.events.bucket import AgentData
        reading = SimpleNamespace(crp=8.5, esr=30.0, payload={})
        profile = SimpleNamespace(age=45, sex=2, bmi=27.0)
        out = build_biomarker_input(reading, profile)
        res = BiomarkerAdapter(BiomarkerAgent()).run(
            AgentData("agent1_biomarker", out), bucket_end=datetime.now(timezone.utc))
        assert res.ok
        assert res.output.vector_dim == 7


# --------------------------------------------------------------------------- #
# Dietary builder (synthetic cache fixtures)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not DENSITY.exists(), reason="dietary cache fixtures not present")
class TestDietaryBuilder:
    def _pipeline(self):
        return DietaryPipeline(DENSITY, FOOD_INDEX)

    def test_builds_rollup_from_meals(self):
        meals = [
            SimpleNamespace(description="oatmeal with banana",
                            meal_at=datetime(2026, 5, 27, 8, 0, tzinfo=timezone.utc)),
            SimpleNamespace(description="chicken and rice",
                            meal_at=datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)),
        ]
        rollup = self._pipeline().build_rollup(meals, "2026-05-27", age=45, sex=2, bmi=27.0)
        assert rollup.meal_count == 2
        assert isinstance(rollup.dii_score, float)
        assert rollup.date == "2026-05-27"

    def test_empty_meals(self):
        rollup = self._pipeline().build_rollup([], "2026-05-27")
        assert rollup.meal_count == 0

    def test_skips_meals_without_description(self):
        meals = [
            SimpleNamespace(description=None, meal_at=None),
            SimpleNamespace(description="apple", meal_at=None),
        ]
        rollup = self._pipeline().build_rollup(meals, "2026-05-27")
        assert rollup.meal_count == 1

    def test_feeds_real_agent(self):
        from immunosense.agents.dietary.agent import DietaryAgent
        from immunosense.adapters import DietaryAdapter
        from immunosense.events.bucket import AgentData
        meals = [SimpleNamespace(description="chicken and rice",
                                 meal_at=datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc))]
        rollup = self._pipeline().build_rollup(meals, "2026-05-27", age=45, sex=2, bmi=27.0)
        res = DietaryAdapter(DietaryAgent()).run(
            AgentData("agent2_dietary", rollup), bucket_end=datetime.now(timezone.utc))
        assert res.ok
        assert res.output.vector_dim == 10
