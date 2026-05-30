"""Layer A core types — the immutable event vocabulary of ImmunoSense.

This module defines the foundational data structures for the event log
(Challenge 1, Layer A). Everything that happens in the system — an agent
producing output, an agent failing, a patient pressing the flare button,
the Conductor evaluating a bucket — is recorded as an immutable Event.

Design principles (locked in the architecture design session):
    1. Events are immutable (frozen dataclass). Layer A is append-only.
    2. Every event carries a trace_id so one bucket evaluation can be
       followed across all 5 agents + fusion + decision.
    3. Events are JSON-serializable (NDJSON storage). The payload is a
       plain dict; no numpy arrays or custom objects leak into Layer A.
    4. schema_version is stamped on every event so the log format can
       evolve without breaking historical reads.

These types are shared infrastructure. The Conductor, MEM0 (Sprint 7), and
the Auto-Research Loop (Sprint 9) all consume Layer A events. None of them
owns this module — it sits at the top level of the package.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "v1"


class EventType(str, Enum):
    """The kinds of events Layer A records.

    Inherits from str so values serialize to plain strings in JSON without
    custom encoders (``EventType.AGENT_OUTPUT == "agent_output"``).
    """

    # An agent successfully produced output for a bucket.
    AGENT_OUTPUT = "agent_output"
    # An agent's process() failed; payload carries the error detail.
    AGENT_ERROR = "agent_error"
    # The patient pressed the flare button (critical-event override).
    FLARE_BUTTON = "flare_button"
    # The Conductor finished evaluating a bucket (records the report summary).
    BUCKET_EVAL = "bucket_eval"
    # A clinical event was logged (lab result, medication change, visit).
    CLINICAL_EVENT = "clinical_event"


class ConfidenceLevel(str, Enum):
    """The four confidence levels from Challenge 7 (missing data handling).

    Aggregated from per-agent quality scores. INSUFFICIENT means the
    Conductor should suppress probability display — there isn't enough
    evidence to make a trustworthy statement.
    """

    INSUFFICIENT = "insufficient"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Numeric ordering for comparisons (INSUFFICIENT=0 .. HIGH=3)."""
        return {
            ConfidenceLevel.INSUFFICIENT: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MODERATE: 2,
            ConfidenceLevel.HIGH: 3,
        }[self]


def new_trace_id(prefix: str = "trace") -> str:
    """Generate a unique trace ID for following work through the system.

    Mirrors the pattern in BaseAgent._new_trace_id so trace IDs look
    consistent whether they originate in an agent or in the Conductor.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utc_now() -> datetime:
    """Timezone-aware current UTC timestamp. Single source for 'now'."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Event:
    """One immutable record in Layer A.

    Fields:
        event_id: Unique UUID for this event.
        user_id: Which patient this event belongs to.
        timestamp: When the event occurred (UTC, ISO-8601 on disk).
        bucket_id: The 6h bucket this event falls in
            (e.g. "patient001_2026-05-27_T2"). See bucket.py.
        event_type: One of EventType.
        agent_id: Which agent produced this (None for non-agent events
            like FLARE_BUTTON or BUCKET_EVAL).
        payload: JSON-serializable dict carrying the event's content.
            For AGENT_OUTPUT this is a serialized summary of AgentOutput
            (NOT the raw numpy vector — see to_payload helpers).
        quality: 0..1 quality/confidence associated with this event
            (agent self-reported confidence for AGENT_OUTPUT events).
        trace_id: Links this event to one logical unit of work.
        schema_version: Format version for forward compatibility.
    """

    event_id: str
    user_id: str
    timestamp: datetime
    bucket_id: str
    event_type: EventType
    agent_id: Optional[str]
    payload: dict
    quality: float = 0.0
    trace_id: str = ""
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def create(
        user_id: str,
        bucket_id: str,
        event_type: EventType,
        payload: dict,
        agent_id: Optional[str] = None,
        quality: float = 0.0,
        trace_id: str = "",
        timestamp: Optional[datetime] = None,
    ) -> "Event":
        """Build an Event with sensible defaults (UUID, now, trace).

        This is the normal way to make an event. The frozen constructor
        is still available for deserialization (see from_dict).
        """
        return Event(
            event_id=uuid.uuid4().hex,
            user_id=user_id,
            timestamp=timestamp or utc_now(),
            bucket_id=bucket_id,
            event_type=EventType(event_type),
            agent_id=agent_id,
            payload=payload,
            quality=float(quality),
            trace_id=trace_id or new_trace_id(),
            schema_version=SCHEMA_VERSION,
        )

    # ------------------------------------------------------------------ #
    # Serialization (NDJSON: one Event == one JSON object == one line)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict (timestamp -> ISO string)."""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "bucket_id": self.bucket_id,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "payload": self.payload,
            "quality": self.quality,
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Serialize to a single JSON line (no embedded newlines)."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "Event":
        """Reconstruct an Event from a parsed dict (inverse of to_dict)."""
        ts = d["timestamp"]
        timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
        return Event(
            event_id=d["event_id"],
            user_id=d["user_id"],
            timestamp=timestamp,
            bucket_id=d["bucket_id"],
            event_type=EventType(d["event_type"]),
            agent_id=d.get("agent_id"),
            payload=d.get("payload", {}),
            quality=float(d.get("quality", 0.0)),
            trace_id=d.get("trace_id", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    @staticmethod
    def from_json(line: str) -> "Event":
        """Parse one NDJSON line into an Event."""
        return Event.from_dict(json.loads(line))
