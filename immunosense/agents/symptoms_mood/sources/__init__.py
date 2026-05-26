"""Symptom data sources.

Sources:
    MockSymptomSource         — synthetic data sampled from disease norms
    VoiceTranscriptSource     — Claude API extraction from voice transcripts
    StructuredFormSource      — direct slider/checkbox input
    FlareButtonSource         — explicit "I'm having a flare" signal
    CompositeSymptomSource    — merges all of the above with provenance
"""

from immunosense.agents.symptoms_mood.sources.composite import CompositeSymptomSource
from immunosense.agents.symptoms_mood.sources.flare_button import FlareButtonSource
from immunosense.agents.symptoms_mood.sources.form import StructuredFormSource
from immunosense.agents.symptoms_mood.sources.mock import MockSymptomSource
from immunosense.agents.symptoms_mood.sources.voice import VoiceTranscriptSource

__all__ = [
    "CompositeSymptomSource",
    "FlareButtonSource",
    "StructuredFormSource",
    "MockSymptomSource",
    "VoiceTranscriptSource",
]
