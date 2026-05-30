"""Layer A — the immutable event log and temporal grid (Challenge 1).

Shared infrastructure consumed by the Conductor, MEM0, and Auto-Research.
"""

from immunosense.events.bucket import (
    BUCKET_HOURS,
    BUCKETS_PER_DAY,
    AgentData,
    BucketBuilder,
    UserBucket,
    TimeBucket,
    freshness_weight,
)
from immunosense.events.event_log import EventLog
from immunosense.events.types import (
    SCHEMA_VERSION,
    ConfidenceLevel,
    Event,
    EventType,
    new_trace_id,
    utc_now,
)

__all__ = [
    # types
    "Event",
    "EventType",
    "ConfidenceLevel",
    "SCHEMA_VERSION",
    "new_trace_id",
    "utc_now",
    # bucket
    "TimeBucket",
    "BucketBuilder",
    "UserBucket",
    "AgentData",
    "freshness_weight",
    "BUCKET_HOURS",
    "BUCKETS_PER_DAY",
    # log
    "EventLog",
]
