"""Time bucketing and the per-bucket patient data carrier.

Two distinct concerns live here:

1. TimeBucket / BucketBuilder — the temporal grid (Challenge 1).
   The global timeline is divided into fixed 6-hour UTC buckets:
       T0 = 00:00-06:00   T1 = 06:00-12:00
       T2 = 12:00-18:00   T3 = 18:00-24:00
   A bucket_id looks like "patient001_2026-05-27_T2". This deterministic
   id is what every Event carries, so all events in the same 6h window
   group together regardless of which agent produced them.

   v1 uses UTC boundaries (simple, testable, globally consistent).
   Patient-local-timezone bucketing is a documented v2 option.

2. UserBucket — the Option B data carrier.
   The caller (ingestion / app / test harness) builds each agent's domain
   object using that agent's existing Layer 2 pipeline, then drops them
   into a UserBucket. The Conductor routes each domain object to the
   right adapter. Adapters stay thin: they do NOT build domain objects.

   Agents that have no data for a bucket are simply absent from the
   UserBucket — the Conductor's quality scoring handles the gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

BUCKET_HOURS = 6
BUCKETS_PER_DAY = 24 // BUCKET_HOURS  # 4

# Maps the bucket index (0..3) to its label.
_BUCKET_LABELS = {0: "T0", 1: "T1", 2: "T2", 3: "T3"}


@dataclass(frozen=True)
class TimeBucket:
    """One 6-hour window on the UTC grid for a specific patient.

    Fields:
        user_id: Owner of this bucket.
        date: Calendar date (UTC) of the bucket's start.
        index: 0..3 (T0..T3).
        start: Inclusive UTC start of the window.
        end: Exclusive UTC end of the window.
    """

    user_id: str
    date: str  # "YYYY-MM-DD" (UTC)
    index: int
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        """The bucket label, e.g. 'T2'."""
        return _BUCKET_LABELS[self.index]

    @property
    def bucket_id(self) -> str:
        """Stable id used on every Event, e.g. 'patient001_2026-05-27_T2'."""
        return f"{self.user_id}_{self.date}_{self.label}"

    def contains(self, ts: datetime) -> bool:
        """True if timestamp ts falls within [start, end)."""
        ts = _ensure_utc(ts)
        return self.start <= ts < self.end


class BucketBuilder:
    """Builds TimeBuckets from timestamps on the fixed UTC 6h grid."""

    @staticmethod
    def bucket_for(user_id: str, ts: datetime) -> TimeBucket:
        """Return the TimeBucket that timestamp ts falls into."""
        ts = _ensure_utc(ts)
        index = ts.hour // BUCKET_HOURS
        start = ts.replace(
            hour=index * BUCKET_HOURS, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(hours=BUCKET_HOURS)
        return TimeBucket(
            user_id=user_id,
            date=start.strftime("%Y-%m-%d"),
            index=index,
            start=start,
            end=end,
        )

    @staticmethod
    def current_bucket(user_id: str, now: Optional[datetime] = None) -> TimeBucket:
        """Return the bucket containing 'now' (defaults to current UTC)."""
        return BucketBuilder.bucket_for(
            user_id, now or datetime.now(timezone.utc)
        )

    @staticmethod
    def from_bucket_id(bucket_id: str) -> TimeBucket:
        """Reconstruct a TimeBucket from its id string.

        Inverse of TimeBucket.bucket_id. Splits on the LAST two underscores
        so patient ids that themselves contain underscores still parse
        (e.g. "patient_001_2026-05-27_T2").
        """
        try:
            rest, label = bucket_id.rsplit("_", 1)
            user_id, date = rest.rsplit("_", 1)
            index = {v: k for k, v in _BUCKET_LABELS.items()}[label]
        except (ValueError, KeyError) as e:
            raise ValueError(f"Malformed bucket_id: {bucket_id!r}") from e

        start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start = start + timedelta(hours=index * BUCKET_HOURS)
        end = start + timedelta(hours=BUCKET_HOURS)
        return TimeBucket(
            user_id=user_id, date=date, index=index, start=start, end=end
        )


@dataclass
class AgentData:
    """One agent's domain object for a bucket, plus freshness metadata.

    The caller builds `domain_object` using the agent's own Layer 2 pipeline
    (Option B). `produced_at` is when the underlying data was actually
    measured/logged — used by the quality scorer to compute a freshness
    penalty relative to the agent's poll_frequency (Challenge 7).

    Fields:
        agent_id: Which agent this data feeds.
        domain_object: The agent-specific input object (DailyRollup,
            DailyEnvironmentSummary, DailySymptomMoodSummary, a dict of raw
            wearable arrays, a biomarker {demographics, reading} dict, etc.).
            The adapter knows how to shape this into the agent's input_data.
        produced_at: UTC timestamp of when the data was measured/logged.
            Defaults to None, meaning "treat as current" (no staleness).
        extras: Optional side inputs (e.g. flare events) the adapter may pass
            through to the agent's process() call.
    """

    agent_id: str
    domain_object: Any
    produced_at: Optional[datetime] = None
    extras: dict = field(default_factory=dict)


@dataclass
class UserBucket:
    """All available agent data for one patient in one 6h bucket (Option B).

    The caller assembles this. Agents with no data this bucket are simply
    absent from `agent_data`. The Conductor iterates whatever is present,
    routes each AgentData to its adapter, and lets the quality layer handle
    the agents that didn't report.

    Fields:
        bucket: The TimeBucket this data belongs to.
        agent_data: Map agent_id -> AgentData for agents that have data.
        flare_button: Optional (severity) if the patient pressed the flare
            button in this bucket; triggers critical-event handling.
        clinical_events: Optional list of clinical event dicts (labs, visits).
    """

    bucket: TimeBucket
    agent_data: dict = field(default_factory=dict)
    flare_button: Optional[float] = None
    clinical_events: list = field(default_factory=list)

    @property
    def user_id(self) -> str:
        return self.bucket.user_id

    @property
    def bucket_id(self) -> str:
        return self.bucket.bucket_id

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self.agent_data

    def get(self, agent_id: str) -> Optional[AgentData]:
        return self.agent_data.get(agent_id)

    def add(self, agent_data: AgentData) -> None:
        """Register one agent's data for this bucket."""
        self.agent_data[agent_data.agent_id] = agent_data

    @property
    def reporting_agents(self) -> list:
        """agent_ids that have data this bucket, in sorted order."""
        return sorted(self.agent_data.keys())


# ---------------------------------------------------------------------- #
# Freshness — Challenge 7 made concrete by the Sprint 5 audit.
# A reading's staleness is judged RELATIVE to its agent's cadence.
# ---------------------------------------------------------------------- #

# Half-life (in hours) for each poll_frequency string the agents declare.
# A reading exactly one half-life old keeps 50% of its freshness weight.
_FRESHNESS_HALFLIFE_HOURS = {
    "1hr": 6.0,        # wearable — stale within hours
    "6hr": 18.0,       # environment
    "daily": 36.0,     # dietary, symptoms_mood — ~1.5 days
    "weekly": 240.0,   # biomarker — ~10 days
}
_DEFAULT_HALFLIFE_HOURS = 36.0


def freshness_weight(
    poll_frequency: str,
    produced_at: Optional[datetime],
    reference: datetime,
) -> float:
    """Compute a 0..1 freshness weight for a reading.

    Uses exponential decay with a half-life keyed to the agent's
    poll_frequency, so a 6-day-old weekly biomarker reading is still
    fresh while a 6-day-old hourly wearable reading is not.

    Args:
        poll_frequency: The agent's declared cadence ("1hr".."weekly").
        produced_at: When the data was measured/logged (UTC). If None,
            the reading is treated as current (weight 1.0).
        reference: The "now" against which age is measured (bucket end).

    Returns:
        Weight in [0, 1]. 1.0 = perfectly fresh, decaying toward 0.
    """
    if produced_at is None:
        return 1.0
    produced_at = _ensure_utc(produced_at)
    reference = _ensure_utc(reference)
    age_hours = max(0.0, (reference - produced_at).total_seconds() / 3600.0)
    halflife = _FRESHNESS_HALFLIFE_HOURS.get(poll_frequency, _DEFAULT_HALFLIFE_HOURS)
    # Exponential decay: weight = 0.5 ** (age / halflife)
    return float(math.pow(0.5, age_hours / halflife))


def _ensure_utc(ts: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
