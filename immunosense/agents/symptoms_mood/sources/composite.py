"""CompositeSymptomSource - merges voice + form + flare button + mock fallback.

Source priority (lower wins, written last takes precedence):
    1. Voice extraction (Claude Haiku)
    2. Structured form (overrides voice)
    3. Flare button (explicit override signal)
    4. Mock fallback (only fills in features that are still None)
"""

from __future__ import annotations

from typing import Optional

from immunosense.agents.symptoms_mood.sources.flare_button import FlareButtonSource
from immunosense.agents.symptoms_mood.sources.form import StructuredFormSource
from immunosense.agents.symptoms_mood.sources.mock import MockSymptomSource
from immunosense.agents.symptoms_mood.sources.voice import VoiceTranscriptSource
from immunosense.agents.symptoms_mood.types import ALL_FEATURES, FetchedSymptoms


class CompositeSymptomSource:
    """Merges voice + form + flare button + optional mock fallback.

    Args:
        voice: Optional VoiceTranscriptSource. If None, voice extraction is skipped.
        structured: Optional StructuredFormSource. Default instance used if None.
        flare_button: Optional FlareButtonSource. Default instance used if None.
        mock_fallback: Optional MockSymptomSource. If provided, fills in missing
            features at the end with synthetic values.
    """

    def __init__(
        self,
        voice: Optional[VoiceTranscriptSource] = None,
        structured: Optional[StructuredFormSource] = None,
        flare_button: Optional[FlareButtonSource] = None,
        mock_fallback: Optional[MockSymptomSource] = None,
    ) -> None:
        self.voice = voice
        self.structured = structured or StructuredFormSource()
        self.flare_button = flare_button or FlareButtonSource()
        self.mock_fallback = mock_fallback

    def assemble(
        self,
        patient_id: str,
        target_date: str,
        transcript: Optional[str] = None,
        form_data: Optional[dict] = None,
        flare_event_severity: Optional[float] = None,
    ) -> FetchedSymptoms:
        """Assemble a merged FetchedSymptoms from all available sources."""
        merged = FetchedSymptoms()

        # 1. Voice extraction (if transcript provided AND voice source configured)
        if transcript and self.voice is not None:
            voice_result = self.voice.extract(transcript, patient_id, target_date)
            for feat in ALL_FEATURES:
                v = getattr(voice_result, feat)
                if v is not None:
                    setattr(merged, feat, v)
                    merged.confidence[feat] = "voice_extracted"
                    merged.sources[feat] = "claude-haiku"
            merged.emotional_valence = voice_result.emotional_valence
            merged.new_symptom_mentions = list(voice_result.new_symptom_mentions)
            merged.errors.extend(voice_result.errors)
            if voice_result.explicit_flare:
                merged.explicit_flare = True

        # 2. Structured form (overrides voice for any provided fields)
        if form_data:
            structured_result = self.structured.from_dict(form_data)
            for feat in ALL_FEATURES:
                v = getattr(structured_result, feat)
                if v is not None:
                    setattr(merged, feat, v)
                    merged.confidence[feat] = "structured"
                    merged.sources[feat] = "form"
            if structured_result.explicit_flare:
                merged.explicit_flare = True
                if structured_result.explicit_flare_severity is not None:
                    merged.explicit_flare_severity = (
                        structured_result.explicit_flare_severity
                    )

        # 3. Flare button (explicit override)
        if flare_event_severity is not None:
            button_result = self.flare_button.from_event(severity=flare_event_severity)
            merged.explicit_flare = True
            merged.explicit_flare_severity = max(
                merged.explicit_flare_severity or 0.0,
                button_result.explicit_flare_severity or 0.0,
            )
            merged.confidence["explicit_flare"] = "flare_button"
            merged.sources["explicit_flare"] = "patient_button"

        # 4. Mock fallback (fills only what's still missing)
        if self.mock_fallback is not None:
            mock_result = self.mock_fallback.fetch(patient_id, target_date)
            for feat in ALL_FEATURES:
                if getattr(merged, feat) is None:
                    setattr(merged, feat, getattr(mock_result, feat))
                    merged.confidence[feat] = "synthetic"
                    merged.sources[feat] = "mock"

        return merged
