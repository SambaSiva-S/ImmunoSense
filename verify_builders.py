"""Verify the three Phase 1 builders.

Module-level imports (same discipline as the other verifiers). Symptom and
biomarker are checked fully. Dietary is checked using the small synthetic cache
fixtures (server/tests/fixtures/), so this verifier does NOT require the real
NHANES files — run build_dietary_caches.py separately for the real caches.

Run from project root:
    venv\\Scripts\\python.exe verify_builders.py
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from server.builders import (
    build_biomarker_input,
    build_symptom_summary,
    DietaryPipeline,
    build_caches,
)

# real agents (to confirm builders produce consumable domain objects)
from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent
from immunosense.agents.biomarker.agent import BiomarkerAgent
from immunosense.agents.dietary.agent import DietaryAgent
from immunosense.adapters import SymptomsMoodAdapter, BiomarkerAdapter, DietaryAdapter
from immunosense.events.bucket import AgentData

FIXTURES = Path(__file__).parent / "server" / "tests" / "fixtures"


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("Builders verification")
    print("=" * 50)
    now = datetime.now(timezone.utc)

    print("\n1. Symptom builder (raw logs -> DailySymptomMoodSummary)")
    logs = [
        SimpleNamespace(fatigue=5.0, joint_pain=4.0, brain_fog_severity=None,
                        gi_distress=None, skin_severity=None, sleep_severity=None,
                        energy_severity=None, wellness_severity=None,
                        phq8_score=8.0, gad7_score=None, free_text=None, source="tap"),
        SimpleNamespace(fatigue=7.0, joint_pain=None, brain_fog_severity=None,
                        gi_distress=3.0, skin_severity=None, sleep_severity=None,
                        energy_severity=None, wellness_severity=None,
                        phq8_score=None, gad7_score=None,
                        free_text="rough night", source="voice"),
    ]
    s = build_symptom_summary(logs, "2026-05-27", "u1", "SLE")
    check("max-per-field fold (fatigue 5,7 -> 7)", s.fatigue == 7.0)
    check("field only in one log preserved (gi_distress=3)", s.gi_distress == 3.0)
    check("free text collected", "rough night" in s.new_symptom_mentions)
    check("sources tracked", s.sources == {"tap": 1, "voice": 1})
    res = SymptomsMoodAdapter(SymptomsMoodAgent()).run(
        AgentData("agent5_symptoms_mood", s), bucket_end=now)
    check("feeds real symptoms agent (dim 36)", res.ok and res.output.vector_dim == 36)

    print("\n2. Biomarker builder (reading + profile -> {demographics, reading})")
    reading = SimpleNamespace(crp=8.5, esr=30.0, payload={"WBC": 7.2})
    profile = SimpleNamespace(age=45, sex=2, bmi=27.0)
    out = build_biomarker_input(reading, profile)
    check("demographics assembled", out["demographics"] == {"age": 45.0, "sex": 2, "bmi": 27.0})
    check("CRP/ESR mapped", out["reading"]["CRP"] == 8.5 and out["reading"]["ESR"] == 30.0)
    check("payload merged into reading", out["reading"]["WBC"] == 7.2)
    res2 = BiomarkerAdapter(BiomarkerAgent()).run(
        AgentData("agent1_biomarker", out), bucket_end=now)
    check("feeds real biomarker agent (dim 7)", res2.ok and res2.output.vector_dim == 7)

    print("\n3. Dietary builder (meals -> DailyRollup via NHANES pipeline)")
    density = FIXTURES / "density_small.pkl"
    food_index = FIXTURES / "food_index_small.pkl"
    if density.exists() and food_index.exists():
        pipeline = DietaryPipeline(density, food_index)
        meals = [
            SimpleNamespace(description="chicken and rice",
                            meal_at=datetime(2026, 5, 27, 13, 0, tzinfo=timezone.utc)),
            SimpleNamespace(description="oatmeal with banana",
                            meal_at=datetime(2026, 5, 27, 8, 0, tzinfo=timezone.utc)),
        ]
        rollup = pipeline.build_rollup(meals, "2026-05-27", age=45, sex=2, bmi=27.0)
        check("rollup built from meals (count 2)", rollup.meal_count == 2)
        check("dii_score is a float", isinstance(rollup.dii_score, float))
        res3 = DietaryAdapter(DietaryAgent()).run(
            AgentData("agent2_dietary", rollup), bucket_end=now)
        check("feeds real dietary agent (dim 10)", res3.ok and res3.output.vector_dim == 10)
    else:
        check("dietary fixtures present", False)
        print("    (fixtures missing — cannot verify dietary builder)")

    print("\n4. Cache builder available")
    check("build_caches importable", callable(build_caches))

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Builders verified OK.")
        print("\nNote: dietary verified with synthetic fixtures. For real food")
        print("matching, run build_dietary_caches.py against your NHANES .XPT files.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
