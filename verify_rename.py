"""Verify the user_id rename applied correctly.

Module-level imports (same discipline as verify_sprint5/6). Confirms:
  - the renamed symbols exist (UserBucket, validate_user_bucket)
  - Event/bucket use user_id, not patient_id
  - the old names are gone from infrastructure
  - a minimal end-to-end evaluation still works

Run from project root:
    venv\\Scripts\\python.exe verify_rename.py
"""

# Renamed infrastructure symbols
from immunosense.events import (
    Event,
    EventType,
    BucketBuilder,
    UserBucket,          # renamed from PatientBucket
    AgentData,
    EventLog,
)
from immunosense.conductor.utils import validate_user_bucket  # renamed
from immunosense.conductor import Conductor
from immunosense.adapters import AdapterRegistry, SymptomsMoodAdapter
from immunosense.agents.base import AgentOutput, BaseAgent

import inspect
import tempfile
from datetime import datetime, timezone

import numpy as np


class _FakeAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_id = "agent5_symptoms_mood"
        self.output_dim = 36
        self.poll_frequency = "daily"

    def process(self, input_data):
        return AgentOutput(
            agent_id=self.agent_id, timestamp=datetime.now(timezone.utc),
            data={"ok": True}, vector=np.ones(36) * 0.5, vector_dim=36,
            alerts=[], confidence=0.8,
        )


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("user_id rename verification")
    print("=" * 50)

    print("\n1. Renamed symbols exist")
    check("UserBucket importable", UserBucket is not None)
    check("validate_user_bucket importable", validate_user_bucket is not None)

    print("\n2. Event uses user_id (not patient_id)")
    ev = Event.create(
        user_id="u1",
        bucket_id="u1_2026-05-27_T2",
        event_type=EventType.AGENT_OUTPUT,
        payload={"x": 1},
    )
    check("Event.user_id field present", hasattr(ev, "user_id"))
    check("Event has NO patient_id field", not hasattr(ev, "patient_id"))
    check("Event.create accepts user_id kwarg",
          "user_id" in inspect.signature(Event.create).parameters)
    d = ev.to_dict()
    check("serialized dict uses 'user_id' key", "user_id" in d and "patient_id" not in d)

    print("\n3. Bucket + UserBucket use user_id")
    b = BucketBuilder.bucket_for("u1", datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc))
    check("TimeBucket.user_id", hasattr(b, "user_id") and b.user_id == "u1")
    ub = UserBucket(bucket=b)
    check("UserBucket.user_id", ub.user_id == "u1")

    print("\n4. End-to-end still works with user_id")
    registry = AdapterRegistry()
    registry.register(SymptomsMoodAdapter(_FakeAgent()))
    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(tmp)
        conductor = Conductor(registry=registry, event_log=log, disease="SLE")
        ub.add(AgentData("agent5_symptoms_mood", "summary",
                         produced_at=datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)))
        report = conductor.evaluate_bucket(ub)
        check("evaluate_bucket returns a report", report is not None)
        check("report.user_id is set", getattr(report, "user_id", None) == "u1")
        events = log.read_bucket("u1", b.bucket_id)
        check("Layer A events written under user_id", len(events) > 0)
        check("all events carry user_id",
              all(e.user_id == "u1" for e in events))

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Rename verified OK.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
