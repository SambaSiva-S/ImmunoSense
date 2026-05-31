"""Scheduler runner — evaluate users who are due, per their local timezone.

Design (Phase 1, matches Challenge 9 "no heavy infra"): instead of a precise
long-running timer per user, expose run_due_evaluations() that a simple periodic
trigger (cron, Windows Task Scheduler, a cloud scheduled job, or an APScheduler
interval) calls — e.g. hourly. Each call:

  1. lists users with a profile (active users)
  2. for each, checks is_due() against their timezone + cadence + last run
  3. if due, runs the evaluation and records the run time

This is idempotent within a period: a user won't be re-evaluated until their
next scheduled local slot, even if the poller runs many times in between.

Cadence + local hour are configurable globally (Settings) and can later be made
per-user. Default: once daily at 21:00 local.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from server.api.config import Settings
from server.api.service import EvaluationService
from server.api.tracelog import log
from server.scheduler.timezones import is_due
from server.db.models import Profile


class SchedulerRunner:
    def __init__(self, session_factory, settings: Settings,
                 cadence: str = "daily", local_hour: int = 21):
        self.session_factory = session_factory
        self.settings = settings
        self.cadence = cadence
        self.local_hour = local_hour
        self.service = EvaluationService(session_factory, settings)
        # Explicit per-user last-run tracking, decoupled from the Conductor's
        # internal evaluated_at clock. In-memory for Phase 1; for multi-process
        # deployment this would move to a small scheduler_runs table.
        self._last_runs: dict[str, datetime] = {}

    def _active_users(self) -> list[tuple[str, str]]:
        """Return (user_id, timezone) for users with a profile."""
        with self.session_factory() as s:
            rows = s.execute(select(Profile.user_id, Profile.timezone)).all()
        return [(uid, tz or "UTC") for uid, tz in rows]

    def _last_run(self, user_id: str) -> datetime | None:
        """Most recent scheduled run for a user (scheduler-tracked)."""
        return self._last_runs.get(user_id)

    def run_due_evaluations(self, now_utc: datetime | None = None) -> dict:
        """Evaluate all users currently due. Returns a summary."""
        now_utc = now_utc or datetime.now(timezone.utc)
        evaluated, skipped, errored = [], [], []

        for user_id, tz_name in self._active_users():
            try:
                due = is_due(
                    tz_name, self._last_run(user_id),
                    cadence=self.cadence, local_hour=self.local_hour, now_utc=now_utc,
                )
                if not due:
                    skipped.append(user_id)
                    continue
                self.service.evaluate(user_id, now_utc)
                self._last_runs[user_id] = now_utc
                evaluated.append(user_id)
            except Exception as exc:  # noqa: BLE001 — one user's failure mustn't stop the rest
                log.error(f"scheduled eval failed for {user_id}: {type(exc).__name__}: {exc}")
                errored.append(user_id)

        summary = {
            "ran_at": now_utc.isoformat(),
            "cadence": self.cadence,
            "evaluated": len(evaluated),
            "skipped": len(skipped),
            "errored": len(errored),
        }
        log.info(f"scheduler run: {summary}")
        return summary
