"""Agent 5 — Symptoms & Mood Agent.

Captures the patient's subjective experience: symptoms, mood, energy, function.
The only agent whose data comes from the patient themselves.

Special role in the system:
    1. Canonical source of flare events. Agent 5 produces daily flare_score [0-1]
       that the Conductor distributes to other agents as their observe_flare() input.
    2. JEPA modality. Emits a 36-dim dense embedding per day for the World Model.
    3. Closest thing to ground truth. Patient-reported symptoms are the most
       direct measure of disease state.

Public API::

    >>> from immunosense.agents.symptoms_mood import (
    ...     SymptomsMoodAgent, process_symptom_day, MockSymptomSource
    ... )
    >>> agent = SymptomsMoodAgent(patient_id='p001')
    >>> source = MockSymptomSource(disease='RA')
    >>> summary = process_symptom_day(
    ...     'p001', '2026-04-01', disease='RA',
    ...     composite_source=CompositeSymptomSource(mock_fallback=source),
    ... )
    >>> agent.observe(summary)
    >>> agent.daily_flare_score()
    0.41
    >>> agent.jepa_embedding().shape
    (36,)
"""

from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent
from immunosense.agents.symptoms_mood.jepa_emit import compute_jepa_embedding
from immunosense.agents.symptoms_mood.memory import (
    HistoricalPattern,
    MemoryStore,
    StubMemoryStore,
)
from immunosense.agents.symptoms_mood.pipeline import (
    compute_daily_flare_score,
    process_symptom_day,
)
from immunosense.agents.symptoms_mood.sources.composite import CompositeSymptomSource
from immunosense.agents.symptoms_mood.sources.flare_button import FlareButtonSource
from immunosense.agents.symptoms_mood.sources.form import StructuredFormSource
from immunosense.agents.symptoms_mood.sources.mock import MockSymptomSource
from immunosense.agents.symptoms_mood.sources.voice import VoiceTranscriptSource
from immunosense.agents.symptoms_mood.types import (
    DailySymptomMoodSummary,
    DetectedSymptomPattern,
    FetchedSymptoms,
    HypothesisEvidence,
    SymptomsMoodAgentReport,
)
from immunosense.agents.symptoms_mood.wellness import compute_wellness_signature

__all__ = [
    # Main agent
    "SymptomsMoodAgent",
    # Pipeline
    "process_symptom_day",
    "compute_daily_flare_score",
    "compute_jepa_embedding",
    "compute_wellness_signature",
    # Sources
    "CompositeSymptomSource",
    "MockSymptomSource",
    "VoiceTranscriptSource",
    "StructuredFormSource",
    "FlareButtonSource",
    # Memory
    "MemoryStore",
    "StubMemoryStore",
    "HistoricalPattern",
    # Types
    "DailySymptomMoodSummary",
    "DetectedSymptomPattern",
    "FetchedSymptoms",
    "HypothesisEvidence",
    "SymptomsMoodAgentReport",
]
