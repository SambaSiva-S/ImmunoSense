"""Memory store interface for cross-session continuity (Challenge 8 scaffolding).

The MemoryStore Protocol defines the interface Agent 5 (and others) use to
record historical observations and query similar past situations.

In v1, ``StubMemoryStore`` provides an in-memory placeholder. When the real
MEM0 implementation (Challenge 8) lands, it can swap in transparently by
satisfying the same Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from immunosense.agents.symptoms_mood.types import DailySymptomMoodSummary


@dataclass
class HistoricalPattern:
    """A historically-similar pattern, surfaced from long-term memory."""

    pattern_id: str
    similarity: float
    summary: str
    date_range: tuple


class MemoryStore(Protocol):
    """Protocol that any memory store implementation must satisfy."""

    def query_patterns(
        self,
        patient_id: str,
        symptom_cluster: dict,
        top_k: int = 3,
    ) -> list:
        """Return up to top_k HistoricalPattern items most similar to symptom_cluster."""
        ...

    def add_observation(
        self, patient_id: str, summary: "DailySymptomMoodSummary"
    ) -> None:
        """Record one daily observation in long-term memory."""
        ...


class StubMemoryStore:
    """In-memory placeholder for MEM0. Records observations but never returns patterns.

    Use until the real MEM0 implementation (Challenge 8) is available.
    """

    def __init__(self) -> None:
        self._log: dict[str, list] = {}

    def query_patterns(
        self,
        patient_id: str,
        symptom_cluster: dict,
        top_k: int = 3,
    ) -> list:
        """Stub: always returns empty list."""
        return []

    def add_observation(
        self, patient_id: str, summary: "DailySymptomMoodSummary"
    ) -> None:
        """Append observation to in-memory log."""
        self._log.setdefault(patient_id, []).append(summary)

    def n_observations(self, patient_id: str) -> int:
        """Convenience: count observations recorded for a patient."""
        return len(self._log.get(patient_id, []))
