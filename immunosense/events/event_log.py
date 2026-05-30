"""EventLog — the append-only NDJSON store for Layer A.

Storage layout (Challenge 1, locked):
    {root}/{user_id}/{date}.ndjson

One JSON object per line, one line per event. Append-only: events are never
mutated or deleted in normal operation. This is the system's audit trail and
the source of truth that Layer B buckets, MEM0 (Sprint 7), and the
Auto-Research Loop (Sprint 9) all derive from.

Why NDJSON for v1 (locked reasoning):
    - Radical simplicity: no DB to run, inspect by eye, trivial to back up.
    - Append is O(1); one open-append-close per event.
    - Reading a day is one file; reading a range is a handful of files.
Migration path: when a patient exceeds ~hundreds of thousands of events, or
multi-patient queries become a bottleneck, swap this class's implementation
for SQLite/DuckDB/Postgres. The interface stays the same; callers don't change.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from immunosense.events.types import Event, EventType


class EventLog:
    """Append-only NDJSON event store rooted at a directory.

    The store is keyed by (user_id, date). All reads and writes go
    through this class so the on-disk format is an implementation detail.
    """

    def __init__(self, root: Union[str, Path]):
        """Create or attach to an event log rooted at `root`.

        The root directory is created if it doesn't exist. Per-patient and
        per-day files are created lazily on first append.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def _patient_dir(self, user_id: str) -> Path:
        d = self.root / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _file_for(self, user_id: str, date: str) -> Path:
        """Path to the NDJSON file for one patient-day."""
        return self._patient_dir(user_id) / f"{date}.ndjson"

    @staticmethod
    def _date_of(event: Event) -> str:
        """The UTC calendar date an event belongs to (file partition key).

        Uses the bucket_id's embedded date when present so an event always
        lands in the same file as the rest of its bucket, even if its raw
        timestamp is a hair across a midnight boundary.
        """
        parts = event.bucket_id.rsplit("_", 2)
        if len(parts) == 3:
            return parts[1]
        return event.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def append(self, event: Event) -> None:
        """Append one event to its patient-day file."""
        path = self._file_for(event.user_id, self._date_of(event))
        with path.open("a", encoding="utf-8") as f:
            f.write(event.to_json())
            f.write("\n")

    def append_many(self, events: Iterable[Event]) -> int:
        """Append a batch of events. Groups by file to minimize opens.

        Returns the number of events written.
        """
        by_file: dict[Path, list[Event]] = defaultdict(list)
        for ev in events:
            by_file[self._file_for(ev.user_id, self._date_of(ev))].append(ev)

        count = 0
        for path, evs in by_file.items():
            with path.open("a", encoding="utf-8") as f:
                for ev in evs:
                    f.write(ev.to_json())
                    f.write("\n")
                    count += 1
        return count

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def read_day(self, user_id: str, date: str) -> list:
        """Read all events for one patient-day, in file (append) order.

        Returns an empty list if the file doesn't exist.
        """
        path = self._file_for(user_id, date)
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(Event.from_json(line))
        return events

    def read_range(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
    ) -> list:
        """Read all events across an inclusive date range [start, end].

        Dates are "YYYY-MM-DD" strings. Returns events sorted by timestamp.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError(f"end_date {end_date} precedes start_date {start_date}")

        events: list = []
        cursor = start
        while cursor <= end:
            events.extend(self.read_day(user_id, cursor.strftime("%Y-%m-%d")))
            cursor += timedelta(days=1)
        events.sort(key=lambda e: e.timestamp)
        return events

    def read_bucket(self, user_id: str, bucket_id: str) -> list:
        """Read all events tagged with a specific bucket_id."""
        # The date is embedded in the bucket_id; read that day and filter.
        parts = bucket_id.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed bucket_id: {bucket_id!r}")
        date = parts[1]
        return [e for e in self.read_day(user_id, date) if e.bucket_id == bucket_id]

    def iter_events(
        self,
        user_id: str,
        event_type: Optional[EventType] = None,
    ) -> Iterator[Event]:
        """Iterate all events for a patient across all days, oldest first.

        Optionally filter by event_type. Streams file-by-file so it does not
        load the entire history into memory at once.
        """
        patient_dir = self.root / user_id
        if not patient_dir.exists():
            return
        for path in sorted(patient_dir.glob("*.ndjson")):
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ev = Event.from_json(line)
                    if event_type is None or ev.event_type == event_type:
                        yield ev

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def count(self, user_id: str) -> int:
        """Total number of events stored for a patient."""
        return sum(1 for _ in self.iter_events(user_id))

    def patients(self) -> list:
        """List user_ids that have any events on disk."""
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())
