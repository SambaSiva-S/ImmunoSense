"""Verify the timezone-aware scheduler.

Module-level imports. Uses in-memory SQLite with users in different timezones to
prove: per-user local due-logic, DST-correct UTC conversion, both cadences,
idempotency, and next-day re-fire.

Run from project root:
    venv\\Scripts\\python.exe verify_scheduler.py
"""

from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.api.config import Settings
from server.db.base import Base
from server.db import models
from server.db.models import Profile, SymptomLog
from server.scheduler import (
    SchedulerRunner,
    is_due,
    local_day_str,
    next_daily_run_utc,
)
from immunosense.events import BucketBuilder


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("Scheduler verification (per-user timezone)")
    print("=" * 50)

    print("\n1. Timezone math (DST-aware)")
    summer = next_daily_run_utc("America/New_York", 21,
                                now_utc=datetime(2026, 7, 1, tzinfo=timezone.utc))
    winter = next_daily_run_utc("America/New_York", 21,
                                now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))
    check("NY 9pm -> UTC differs summer vs winter (DST)", summer.hour != winter.hour)
    check("local_day_str respects timezone",
          local_day_str("America/New_York", datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc)) == "2026-05-30")

    print("\n2. Due-logic per timezone")
    now = datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc)  # NY 21:30, Tokyo 10:30
    check("NY due at its local 9pm", is_due("America/New_York", None, "daily", 21, now) is True)
    check("Tokyo NOT due at its local 10:30am", is_due("Asia/Tokyo", None, "daily", 21, now) is False)
    check("6h_block cadence due at block boundary",
          is_due("UTC", None, "6h_block", now_utc=datetime(2026, 5, 30, 18, 5, tzinfo=timezone.utc)) is True)

    print("\n3. Runner: evaluates only due users")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with sf() as s:
        s.add(Profile(user_id="u_ny", disease="SLE", timezone="America/New_York"))
        s.add(Profile(user_id="u_tokyo", disease="RA", timezone="Asia/Tokyo"))
        s.commit()
        b = BucketBuilder.bucket_for("u_ny", now)
        s.add(SymptomLog(user_id="u_ny", bucket_id=b.bucket_id, logged_at=now, fatigue=6.0))
        s.commit()
    runner = SchedulerRunner(sf, Settings(dev_auth=True), cadence="daily", local_hour=21)

    s1 = runner.run_due_evaluations(now_utc=now)
    check("evaluated 1 (NY), skipped 1 (Tokyo)", s1["evaluated"] == 1 and s1["skipped"] == 1)

    print("\n4. Idempotency + next-day")
    s2 = runner.run_due_evaluations(now_utc=now + timedelta(minutes=30))
    check("no re-eval within same slot", s2["evaluated"] == 0)
    s3 = runner.run_due_evaluations(now_utc=datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc))
    check("re-fires next day", s3["evaluated"] == 1)

    print("\n5. Error isolation")
    check("runner returns summary with errored count", "errored" in s1)

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Scheduler verified OK.")
        print("\nDefault: once daily at 21:00 local. Configure 6h_block cadence via")
        print("run_scheduler.py --cadence 6h_block. Events store in UTC; only")
        print("scheduling + presentation are timezone-aware.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
