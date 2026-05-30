"""PostgresEventLog — the Layer A event store backed by SQLAlchemy/Postgres.

This is the Challenge 8 payoff: it implements the SAME public interface as the
NDJSON EventLog (append / append_many / read_day / read_range / read_bucket /
iter_events / count), so the Conductor consumes it without any change. Swapping
storage from files to Postgres is a one-line construction change at the call
site, exactly as designed.

Every write also emits an audit.access_log row (D10), so PHI access is recorded
from day one. The DB engine is injected, so this works on Postgres (production)
and SQLite (tests) identically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from immunosense.events.types import Event, EventType
from server.db.models import AccessLog, EventRow


class PostgresEventLog:
    """Layer A event store on a relational DB. Drop-in for EventLog."""

    def __init__(self, session_factory, audit: bool = True):
        """Args:
        session_factory: a callable returning a SQLAlchemy Session (e.g. the
            sessionmaker from base.py). A factory (not a single session) so each
            operation is its own short transaction.
        audit: whether to emit audit.access_log rows on writes (default True).
        """
        self._session_factory = session_factory
        self._audit = audit

    # ------------------------------------------------------------------ #
    # Mapping between the domain Event and the EventRow ORM model
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_row(ev: Event) -> EventRow:
        return EventRow(
            event_id=ev.event_id,
            user_id=ev.user_id,
            timestamp=ev.timestamp,
            bucket_id=ev.bucket_id,
            event_type=ev.event_type.value,
            agent_id=ev.agent_id,
            payload=ev.payload,
            quality=ev.quality,
            trace_id=ev.trace_id,
            schema_version=ev.schema_version,
        )

    @staticmethod
    def _to_event(row: EventRow) -> Event:
        return Event(
            event_id=row.event_id,
            user_id=row.user_id,
            timestamp=_aware(row.timestamp),
            bucket_id=row.bucket_id,
            event_type=EventType(row.event_type),
            agent_id=row.agent_id,
            payload=row.payload or {},
            quality=row.quality,
            trace_id=row.trace_id,
            schema_version=row.schema_version,
        )

    def _audit_write(self, session: Session, user_id: str, n: int) -> None:
        if self._audit:
            session.add(
                AccessLog(
                    user_id=user_id,
                    accessor_id=user_id,
                    action="write",
                    resource="health.events",
                )
            )

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def append(self, event: Event) -> None:
        with self._session_factory() as session:
            session.add(self._to_row(event))
            self._audit_write(session, event.user_id, 1)
            session.commit()

    def append_many(self, events: Iterable[Event]) -> int:
        events = list(events)
        if not events:
            return 0
        with self._session_factory() as session:
            session.add_all(self._to_row(e) for e in events)
            # one audit row per distinct user in the batch
            for uid in {e.user_id for e in events}:
                self._audit_write(session, uid, 0)
            session.commit()
        return len(events)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def read_day(self, user_id: str, date: str) -> list:
        """All events for a user on a UTC calendar date (by bucket_id date)."""
        with self._session_factory() as session:
            stmt = (
                select(EventRow)
                .where(EventRow.user_id == user_id)
                .where(EventRow.bucket_id.like(f"%_{date}_%"))
                .order_by(EventRow.timestamp)
            )
            rows = session.execute(stmt).scalars().all()
            return [self._to_event(r) for r in rows]

    def read_range(self, user_id: str, start_date: str, end_date: str) -> list:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError(f"end_date {end_date} precedes start_date {start_date}")
        # Read by the bucket_id's embedded date instead of timestamp comparison.
        # This avoids aware/naive datetime mismatches across engines (SQLite
        # stores naive; Postgres stores aware) and is consistent with how the
        # NDJSON EventLog partitions by date.
        dates = []
        cursor = start
        while cursor <= end:
            dates.append(cursor.strftime("%Y-%m-%d"))
            cursor += timedelta(days=1)
        events: list = []
        for d in dates:
            events.extend(self.read_day(user_id, d))
        events.sort(key=lambda e: e.timestamp)
        return events

    def read_bucket(self, user_id: str, bucket_id: str) -> list:
        with self._session_factory() as session:
            stmt = (
                select(EventRow)
                .where(EventRow.user_id == user_id)
                .where(EventRow.bucket_id == bucket_id)
                .order_by(EventRow.timestamp)
            )
            rows = session.execute(stmt).scalars().all()
            return [self._to_event(r) for r in rows]

    def iter_events(
        self, user_id: str, event_type: Optional[EventType] = None
    ) -> Iterator[Event]:
        with self._session_factory() as session:
            stmt = (
                select(EventRow)
                .where(EventRow.user_id == user_id)
                .order_by(EventRow.timestamp)
            )
            if event_type is not None:
                stmt = stmt.where(EventRow.event_type == event_type.value)
            for row in session.execute(stmt).scalars().all():
                yield self._to_event(row)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def count(self, user_id: str) -> int:
        from sqlalchemy import func

        with self._session_factory() as session:
            stmt = select(func.count()).select_from(EventRow).where(
                EventRow.user_id == user_id
            )
            return int(session.execute(stmt).scalar_one())

    def users(self) -> list:
        with self._session_factory() as session:
            stmt = select(EventRow.user_id).distinct().order_by(EventRow.user_id)
            return [r for (r,) in session.execute(stmt).all()]

    # Back-compat alias for the NDJSON interface's patients() method.
    def patients(self) -> list:
        return self.users()


def _aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC (SQLite returns naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
