"""Timezone-aware scheduling math.

Core principle (locked): events are stored and bucketed in UTC (stable,
unambiguous, what the whole library already uses). Only SCHEDULING and
PRESENTATION are timezone-aware. This module computes, for a given user's
timezone, the next UTC instant at which their scheduled evaluation should fire.

Two cadences (configurable):
  - "daily"   : once per local day at a configured local hour (default 21:00)
  - "6h_block": at the end of each local 6h block (00,06,12,18 local)

DST is handled by zoneinfo: we build the target in local wall-clock time, then
convert to UTC, so the UTC instant shifts correctly across DST boundaries.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_LOCAL_HOUR = 21  # 9pm local — the "reflect on your day" moment
BLOCK_HOURS = 6


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def next_daily_run_utc(tz_name: str, local_hour: int = DEFAULT_LOCAL_HOUR,
                       now_utc: datetime | None = None) -> datetime:
    """Next UTC instant for a once-daily evaluation at `local_hour` local time."""
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = _zone(tz_name)
    now_local = now_utc.astimezone(tz)
    target_local = now_local.replace(hour=local_hour, minute=0, second=0, microsecond=0)
    if target_local <= now_local:
        target_local = target_local + timedelta(days=1)
    return target_local.astimezone(timezone.utc)


def next_block_run_utc(tz_name: str, now_utc: datetime | None = None) -> datetime:
    """Next UTC instant for the end of the current local 6h block (00/06/12/18)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = _zone(tz_name)
    now_local = now_utc.astimezone(tz)
    # Next local boundary hour that is a multiple of BLOCK_HOURS, strictly after now.
    next_block_hour = ((now_local.hour // BLOCK_HOURS) + 1) * BLOCK_HOURS
    base = now_local.replace(minute=0, second=0, microsecond=0)
    if next_block_hour >= 24:
        target_local = (base + timedelta(days=1)).replace(hour=0)
    else:
        target_local = base.replace(hour=next_block_hour)
    return target_local.astimezone(timezone.utc)


def next_run_utc(tz_name: str, cadence: str = "daily",
                 local_hour: int = DEFAULT_LOCAL_HOUR,
                 now_utc: datetime | None = None) -> datetime:
    """Dispatch by cadence. cadence in {"daily", "6h_block"}."""
    if cadence == "6h_block":
        return next_block_run_utc(tz_name, now_utc=now_utc)
    return next_daily_run_utc(tz_name, local_hour=local_hour, now_utc=now_utc)


def local_day_str(tz_name: str, now_utc: datetime | None = None) -> str:
    """The user's current local calendar date (for UI 'today' context)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    return now_utc.astimezone(_zone(tz_name)).strftime("%Y-%m-%d")


def is_due(tz_name: str, last_run_utc: datetime | None, cadence: str = "daily",
           local_hour: int = DEFAULT_LOCAL_HOUR,
           now_utc: datetime | None = None) -> bool:
    """Whether a user is due for evaluation now.

    Due if we've passed their scheduled local time for the current period and
    haven't already run since the previous scheduled instant. This lets a
    simple periodic poller (e.g. hourly) decide per-user without precise timers.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = _zone(tz_name)
    now_local = now_utc.astimezone(tz)

    if cadence == "6h_block":
        # Due at the most recent block boundary at or before now.
        block_hour = (now_local.hour // BLOCK_HOURS) * BLOCK_HOURS
        scheduled_local = now_local.replace(hour=block_hour, minute=0, second=0, microsecond=0)
    else:
        # Due only once today's local target hour has arrived. If it hasn't yet,
        # the user is simply not due (we do NOT reach back to yesterday's slot —
        # a daily scheduler fires when the user crosses the hour, not for missed
        # past slots).
        scheduled_local = now_local.replace(hour=local_hour, minute=0, second=0, microsecond=0)
        if now_local < scheduled_local:
            return False

    scheduled_utc = scheduled_local.astimezone(timezone.utc)
    if now_utc < scheduled_utc:
        return False
    if last_run_utc is None:
        return True
    last = last_run_utc if last_run_utc.tzinfo else last_run_utc.replace(tzinfo=timezone.utc)
    return last < scheduled_utc
