"""Verify the server data layer.

Module-level imports (same discipline as the sprint verifiers). Uses an
in-memory SQLite DB to exercise the exact SQLAlchemy code path that runs on
Postgres/Supabase in production.

Run from the project root:
    venv\\Scripts\\python.exe verify_datalayer.py
"""

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import sessionmaker

from server.db.base import Base, make_engine
from server.db import models
from server.db.event_store import PostgresEventLog
from server.db.models import AccessLog, CalibrationLR, SymptomLog, User
from server.db.seed import seed_reference_data
from server.db.types import EncryptedString

import numpy as np
from datetime import datetime, timezone

from immunosense.events import AgentData, BucketBuilder, EventType, UserBucket
from immunosense.adapters import AdapterRegistry, SymptomsMoodAdapter, WearableAdapter
from immunosense.conductor import Conductor
from immunosense.agents.base import AgentOutput, BaseAgent


class _Elevated(BaseAgent):
    def __init__(self, aid, dim, poll):
        super().__init__()
        self.agent_id = aid
        self.output_dim = dim
        self.poll_frequency = poll

    def process(self, d):
        return AgentOutput(
            agent_id=self.agent_id, timestamp=datetime.now(timezone.utc),
            data={"ok": True}, vector=np.ones(self.output_dim) * 0.8,
            vector_dim=self.output_dim, alerts=[{"severity": "critical"}],
            confidence=0.9,
        )


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("Data layer verification")
    print("=" * 50)

    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    print("\n1. Schema")
    tables = set(inspect(engine).get_table_names())
    check("16 tables created", len([t for t in tables if t != "alembic_version"]) == 16)
    for t in ("users", "events", "symptom_logs", "calibration_lr", "access_log",
              "photos", "consents", "bucket_reports"):
        check(f"table '{t}' exists", t in tables)

    print("\n2. Identity separation (user_id, email isolated)")
    check("SymptomLog has no email column",
          "email" not in SymptomLog.__table__.columns.keys())
    check("SymptomLog keys on user_id",
          "user_id" in SymptomLog.__table__.columns.keys())
    check("email is PHI-typed on User",
          isinstance(User.__table__.c.email.type, EncryptedString))

    print("\n3. PHI encryption seam (Phase 1 passthrough)")
    with SL() as s:
        log = SymptomLog(user_id="u1", bucket_id="u1_2026-05-27_T2",
                         logged_at=datetime.now(timezone.utc),
                         fatigue=7.0, free_text="rough day")
        s.add(log); s.commit(); lid = log.log_id
    with SL() as s:
        got = s.get(SymptomLog, lid)
        check("PHI free_text round-trips (plaintext, Phase 1)", got.free_text == "rough day")

    print("\n4. Reference seeder")
    summary = seed_reference_data(SL)
    check("LR rows seeded (5 agents)", summary["lr_rows"] == 5)
    check("pattern rows seeded (6-8)", 6 <= summary["pattern_rows"] <= 8)
    with SL() as s:
        n_lr = s.execute(select(func.count()).select_from(CalibrationLR)).scalar_one()
    check("calibration_lr populated", n_lr == 5)

    print("\n5. PostgresEventLog drop-in parity (real Conductor)")
    log = PostgresEventLog(SL)
    registry = AdapterRegistry()
    registry.register(SymptomsMoodAdapter(_Elevated("agent5_symptoms_mood", 36, "daily")))
    registry.register(WearableAdapter(_Elevated("agent4_wearable", 29, "1hr")))
    conductor = Conductor(registry=registry, event_log=log, disease="SLE")
    ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
    bucket = BucketBuilder.bucket_for("u_v_1", ts)
    ub = UserBucket(bucket=bucket)
    ub.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
    ub.add(AgentData("agent4_wearable",
                     {"night_df": "df", "rr_intervals": [800], "night_idx": 1}, produced_at=ts))
    report = conductor.evaluate_bucket(ub)
    check("Conductor produced flare_probability", report.flare_probability is not None)
    check("autonomic_stress pattern matched",
          any(p.name == "autonomic_stress" for p in report.matched_patterns))
    events = log.read_bucket("u_v_1", bucket.bucket_id)
    check("events persisted to DB", len(events) == 3)
    check("bucket_eval persisted",
          any(e.event_type == EventType.BUCKET_EVAL for e in events))
    check("single shared trace id", len({e.trace_id for e in events}) == 1)
    check("read_day works", len(log.read_day("u_v_1", "2026-05-27")) == 3)
    check("read_range works", len(log.read_range("u_v_1", "2026-05-26", "2026-05-28")) == 3)
    check("count works", log.count("u_v_1") == 3)
    check("users() works", log.users() == ["u_v_1"])

    print("\n6. Audit logging (D10)")
    with SL() as s:
        n_audit = s.execute(select(func.count()).select_from(AccessLog)).scalar_one()
    check("audit rows written on writes", n_audit > 0)

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Data layer verified OK.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
