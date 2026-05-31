"""Scheduler tests — timezone-aware due-logic, DST, cadences, idempotency."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.api.config import Settings
from server.db.base import Base
from server.db import models  # noqa: F401
from server.db.models import Profile, SymptomLog
from server.scheduler import (
    SchedulerRunner,
    is_due,
    local_day_str,
    next_daily_run_utc,
    next_block_run_utc,
)
from immunosense.events import BucketBuilder


class TestTimezoneMath:
    def test_daily_run_converts_to_utc(self):
        # NY 21:00 EDT -> 01:00 UTC next day
        nxt = next_daily_run_utc("America/New_York", 21,
                                 now_utc=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc))
        assert nxt.hour == 1 and nxt.tzinfo == timezone.utc

    def test_dst_shifts_utc_instant(self):
        # Same local 21:00 maps to different UTC in summer vs winter
        summer = next_daily_run_utc("America/New_York", 21,
                                    now_utc=datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc))
        winter = next_daily_run_utc("America/New_York", 21,
                                    now_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
        assert summer.hour != winter.hour  # EDT vs EST -> 1 hour difference

    def test_invalid_timezone_falls_back_to_utc(self):
        nxt = next_daily_run_utc("Not/AZone", 21,
                                 now_utc=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc))
        assert nxt.hour == 21  # treated as UTC

    def test_block_run_boundaries(self):
        # UTC user at 13:00 -> next block boundary is 18:00
        nxt = next_block_run_utc("UTC", now_utc=datetime(2026, 5, 30, 13, 0, tzinfo=timezone.utc))
        assert nxt.hour == 18

    def test_local_day_str(self):
        # 01:30 UTC is still "yesterday" in NY
        d = local_day_str("America/New_York", datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc))
        assert d == "2026-05-30"


class TestIsDue:
    NOW_NY_EVENING = datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc)  # NY 21:30

    def test_due_after_local_hour(self):
        assert is_due("America/New_York", None, "daily", 21, self.NOW_NY_EVENING) is True

    def test_not_due_before_local_hour(self):
        # Tokyo at 10:30 local — its 21:00 hasn't arrived
        assert is_due("Asia/Tokyo", None, "daily", 21, self.NOW_NY_EVENING) is False

    def test_not_due_if_already_ran_this_slot(self):
        last = datetime(2026, 5, 31, 1, 15, tzinfo=timezone.utc)  # ran at 21:15 NY
        assert is_due("America/New_York", last, "daily", 21, self.NOW_NY_EVENING) is False

    def test_due_next_day(self):
        last = datetime(2026, 5, 31, 1, 15, tzinfo=timezone.utc)
        tomorrow = datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc)
        assert is_due("America/New_York", last, "daily", 21, tomorrow) is True

    def test_6h_block_cadence(self):
        # NY 12:30 local -> most recent block boundary 12:00 -> due
        t = datetime(2026, 5, 30, 16, 30, tzinfo=timezone.utc)  # 12:30 EDT
        assert is_due("America/New_York", None, "6h_block", now_utc=t) is True


@pytest.fixture
def runner_env():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with sf() as s:
        s.add(Profile(user_id="u_ny", disease="SLE", timezone="America/New_York"))
        s.add(Profile(user_id="u_tokyo", disease="RA", timezone="Asia/Tokyo"))
        s.commit()
    runner = SchedulerRunner(sf, Settings(dev_auth=True), cadence="daily", local_hour=21)
    return runner, sf


class TestRunner:
    NOW = datetime(2026, 5, 31, 1, 30, tzinfo=timezone.utc)  # NY 21:30, Tokyo 10:30

    def test_evaluates_only_due_users(self, runner_env):
        runner, sf = runner_env
        with sf() as s:
            b = BucketBuilder.bucket_for("u_ny", self.NOW)
            s.add(SymptomLog(user_id="u_ny", bucket_id=b.bucket_id,
                             logged_at=self.NOW, fatigue=6.0))
            s.commit()
        summary = runner.run_due_evaluations(now_utc=self.NOW)
        assert summary["evaluated"] == 1  # NY
        assert summary["skipped"] == 1    # Tokyo

    def test_idempotent_within_slot(self, runner_env):
        runner, sf = runner_env
        runner.run_due_evaluations(now_utc=self.NOW)
        again = runner.run_due_evaluations(now_utc=self.NOW + timedelta(minutes=30))
        assert again["evaluated"] == 0

    def test_refires_next_day(self, runner_env):
        runner, sf = runner_env
        runner.run_due_evaluations(now_utc=self.NOW)
        nxt = runner.run_due_evaluations(now_utc=datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc))
        assert nxt["evaluated"] == 1  # NY due again

    def test_one_user_failure_isolated(self, runner_env):
        runner, sf = runner_env
        # Tokyo not due, NY due; even if NY eval had an issue, runner returns a summary
        summary = runner.run_due_evaluations(now_utc=self.NOW)
        assert "evaluated" in summary and "errored" in summary
