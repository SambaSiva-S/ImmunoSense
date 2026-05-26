"""Sprint 3 post-apply verification script.

Usage (from C:\\Projects\\immunosense, with venv activated):
    python verify_sprint3.py

Validates that Sprint 3 (Agent 2 Dietary) is correctly installed:
  1. All dietary modules import
  2. DietaryAgent contract (output_dim=10, agent_id, BaseAgent subclass)
  3. End-to-end SyntheticPatient validation (DAIRY trigger detected at lag=2)
  4. Sprint 1+2 still works (Environment, Wearable, Symptoms&Mood imports)

Exit codes:
    0 = all checks passed
    1 = at least one check failed
"""

import sys


def check(label, fn):
    """Run a check, print PASS/FAIL with the label."""
    try:
        fn()
        print(f"  PASS  {label}")
        return True
    except Exception as e:
        print(f"  FAIL  {label}")
        print(f"        {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 70)
    print("Sprint 3 verification")
    print("=" * 70)
    passed = 0
    failed = 0

    def _record(ok):
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1

    # ---- 1. Imports ----
    print("\n[1] Dietary module imports")

    def imp_constants():
        from immunosense.agents.dietary.constants import (
            DII_REF, CONTINUOUS_FEATURES, BOOLEAN_TRIGGERS,
        )
        assert len(DII_REF) == 27
        assert len(CONTINUOUS_FEATURES) == 6
        assert len(BOOLEAN_TRIGGERS) == 4

    def imp_package():
        from immunosense.agents.dietary import (
            DietaryAgent, process_meal, rollup_day,
            MockExtractor, ClaudeHaikuExtractor,
            compute_dii_row, compute_meal_dii,
            OvernightFastTracker, DietaryRobustTracker, DietaryTriggerDetector,
            DailyRollup, MealResult,
        )

    _record(check("dietary.constants", imp_constants))
    _record(check("dietary package facade", imp_package))

    # ---- 2. DietaryAgent contract ----
    print("\n[2] DietaryAgent BaseAgent contract")

    def agent_contract():
        from immunosense.agents.base import BaseAgent
        from immunosense.agents.dietary import DietaryAgent
        assert issubclass(DietaryAgent, BaseAgent)
        assert DietaryAgent.agent_id == "agent2_dietary"
        assert DietaryAgent.output_dim == 10
        agent = DietaryAgent(patient_id="verify_p001")
        assert agent.patient_id == "verify_p001"

    _record(check("agent inherits BaseAgent, output_dim=10", agent_contract))

    # ---- 3. End-to-end SyntheticPatient ----
    print("\n[3] End-to-end SyntheticPatient validation (DAIRY trigger)")

    def e2e_dairy():
        import random as random_mod
        import numpy as np
        import pandas as pd
        from immunosense.agents.dietary import (
            BOOLEAN_TRIGGERS, CONTINUOUS_FEATURES, DailyRollup, DietaryAgent,
        )

        # Generate 60 days with DAIRY as the planted trigger
        rng = random_mod.Random(42)
        np_rng = np.random.RandomState(42)
        start = pd.Timestamp("2026-03-01")
        records = []
        flares = {}

        for i in range(60):
            date = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            rec = {
                "date": date,
                "dii_score": float(np_rng.normal(0.5, 1.0)),
                "omega6_omega3_ratio": float(max(1.0, np_rng.normal(10, 4))),
                "glycemic_load": float(max(20, np_rng.normal(100, 30))),
                "sodium_mg": float(max(500, np_rng.normal(2500, 800))),
                "alcohol_g": float(max(0, np_rng.exponential(5))),
                "gluten_present": rng.random() < 0.55,
                "dairy_present": rng.random() < 0.40,
                "nightshade_present": rng.random() < 0.30,
                "upf_present": rng.random() < 0.25,
                "first_meal_timestamp": f"{date}T08:00:00",
                "last_meal_timestamp": f"{date}T19:00:00",
            }

            flare_prob = 0.70 if rec["dairy_present"] else 0.10
            if rng.random() < flare_prob:
                flare_date = (start + pd.Timedelta(days=i + 2)).strftime("%Y-%m-%d")
                severity = float(np_rng.uniform(1.5, 3.0))
                flares[flare_date] = max(flares.get(flare_date, 0.0), severity)
            records.append(rec)

        agent = DietaryAgent(window=14, n_permutations=500)
        for rec in records:
            rollup = DailyRollup(
                date=rec["date"], meal_count=4, daily_nutrients={},
                dii_score=rec["dii_score"],
                omega6_omega3_ratio=rec["omega6_omega3_ratio"],
                glycemic_load=rec["glycemic_load"],
                sodium_mg=rec["sodium_mg"],
                alcohol_g=rec["alcohol_g"],
                first_meal_timestamp=rec["first_meal_timestamp"],
                last_meal_timestamp=rec["last_meal_timestamp"],
                longest_intraday_gap_hours=float("nan"),
                gluten_present=rec["gluten_present"],
                dairy_present=rec["dairy_present"],
                nightshade_present=rec["nightshade_present"],
                upf_present=rec["upf_present"],
                feature_confidence={"dii_score": "high"},
                daily_dii_percentile=None,
                meal_results=[],
            )
            agent.observe(rollup)
        for date, severity in flares.items():
            agent.observe_flare(date, severity)

        report = agent.analyze()
        # Should detect dairy_present at lag=2
        dairy = [p for p in report.detected_patterns
                 if p.feature == "dairy_present" and p.lag_days == 2]
        assert len(dairy) >= 1, (
            f"Did not detect planted dairy_present at lag=2. "
            f"Got: {[(p.feature, p.lag_days) for p in report.detected_patterns]}"
        )
        assert dairy[0].confidence == "high", (
            f"Expected high confidence; got {dairy[0].confidence}"
        )

    _record(check("DAIRY trigger detected at lag=2 with [high] confidence", e2e_dairy))

    # ---- 4. Sprint 1+2 still works ----
    print("\n[4] Sprint 1+2 imports still working")

    def sprint12_env():
        from immunosense.agents.environment import EnvironmentAgent
        assert EnvironmentAgent.output_dim == 5

    def sprint12_wearable():
        from immunosense.agents.wearable import WearableAgent
        assert WearableAgent.output_dim == 29

    def sprint12_symptoms():
        from immunosense.agents.symptoms_mood import SymptomsMoodAgent
        assert SymptomsMoodAgent.output_dim == 36

    _record(check("Agent 3 (Environment) output_dim=5", sprint12_env))
    _record(check("Agent 4 (Wearable) output_dim=29", sprint12_wearable))
    _record(check("Agent 5 (Symptoms & Mood) output_dim=36", sprint12_symptoms))

    # ---- Summary ----
    print()
    print("=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
