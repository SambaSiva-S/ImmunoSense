"""Verify Sprint 5 installed correctly.

IMPORTANT: all imports are at MODULE LEVEL on purpose. Sprint 4's verify script
wrapped imports inside a function, which produced repeated false-negative
failures (6 rounds) even though direct imports worked fine. Do not move these
imports into a function.

Run from the project root:
    venv\\Scripts\\python.exe verify_sprint5.py
"""

# --- Module-level imports: events ---
from immunosense.events import (
    Event,
    EventType,
    ConfidenceLevel,
    BucketBuilder,
    TimeBucket,
    PatientBucket,
    AgentData,
    EventLog,
    freshness_weight,
    new_trace_id,
)

# --- Module-level imports: adapters ---
from immunosense.adapters import (
    AdapterRegistry,
    BaseAgentAdapter,
    AdapterResult,
    BiomarkerAdapter,
    DietaryAdapter,
    EnvironmentAdapter,
    WearableAdapter,
    SymptomsMoodAdapter,
    ADAPTER_FOR_AGENT,
)

# --- Module-level imports: conductor ---
from immunosense.conductor import Conductor, ConductorReport
from immunosense.conductor.quality import (
    QualityScorer,
    AgentQuality,
    ConfidenceAggregator,
    ConfidenceResult,
)
from immunosense.conductor.fusion import (
    StatisticalFusion,
    Corroboration,
    RiskEngine,
)
from immunosense.conductor.decision import DecisionMaker, Decision
from immunosense.conductor.utils import TraceContext, validate_patient_bucket

# --- Module-level imports: real agents (to confirm wiring) ---
from immunosense.agents.biomarker.agent import BiomarkerAgent
from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent

import tempfile
from datetime import datetime, timezone, timedelta


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("Sprint 5 verification")
    print("=" * 50)

    print("\n1. Imports")
    check("events imports", Event is not None and EventLog is not None)
    check("adapters imports", AdapterRegistry is not None)
    check("conductor imports", Conductor is not None)
    check("quality imports", QualityScorer is not None)
    check("fusion stub imports", StatisticalFusion is not None)
    check("decision stub imports", DecisionMaker is not None)

    print("\n2. Bucketing")
    ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
    b = BucketBuilder.bucket_for("patient001", ts)
    check("bucket id format", b.bucket_id == "patient001_2026-05-27_T2")
    rt = BucketBuilder.from_bucket_id(b.bucket_id)
    check("bucket id round trip", rt.start == b.start)

    print("\n3. Freshness relative to cadence (Challenge 7)")
    ref = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
    old = ref - timedelta(days=6)
    w_weekly = freshness_weight("weekly", old, ref)
    w_hourly = freshness_weight("1hr", old, ref)
    check("6d weekly still fresh (>0.5)", w_weekly > 0.5)
    check("6d hourly stale (<0.01)", w_hourly < 0.01)

    print("\n4. Event log round trip")
    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(tmp)
        ev = Event.create("patient001", b.bucket_id, EventType.AGENT_OUTPUT,
                          {"x": 1}, agent_id="agent5_symptoms_mood", quality=0.8)
        log.append(ev)
        day = log.read_day("patient001", "2026-05-27")
        check("append + read_day", len(day) == 1 and day[0].quality == 0.8)

    print("\n5. End-to-end with real agents")
    bio = BiomarkerAgent()
    sym = SymptomsMoodAgent()
    registry = AdapterRegistry.from_agents([bio, sym])
    check("registry wired 2 agents", len(registry) == 2)

    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(tmp)
        conductor = Conductor(registry=registry, event_log=log)
        pb = PatientBucket(bucket=b)
        # Garbage to biomarker -> must fail gracefully, not crash.
        pb.add(AgentData("agent1_biomarker", {"bad": "data"}, produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        check("evaluate_bucket returns report", isinstance(report, ConductorReport))
        check("failed agent isolated (quality 0)",
              report.agent_quality["agent1_biomarker"].quality == 0.0)
        check("fusion stub returns None", report.flare_probability is None)
        check("corroboration stub returns []", report.matched_patterns == [])
        events = log.read_bucket("patient001", b.bucket_id)
        check("layer A has bucket_eval",
              any(e.event_type == EventType.BUCKET_EVAL for e in events))
        trace_ids = {e.trace_id for e in events}
        check("all events share one trace id", len(trace_ids) == 1)

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Sprint 5 verified OK.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
