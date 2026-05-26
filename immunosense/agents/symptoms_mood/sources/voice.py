"""VoiceTranscriptSource - extract structured symptoms from voice transcripts.

Uses Claude Haiku via the shared anthropic_client. Caches extractions
keyed by transcript hash to avoid duplicate API calls.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

from immunosense.agents.common.anthropic_client import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_MAX_TOKENS,
    get_anthropic_client,
)
from immunosense.agents.symptoms_mood.types import ALL_FEATURES, FetchedSymptoms


_EXTRACTION_PROMPT = """You are a clinical NLP assistant extracting structured symptom data from a patient's voice transcript.

The patient has an autoimmune condition and is describing how they feel today.

Extract values for these features. Use null (not 0) when the feature is not mentioned. Be conservative - do not infer high severities from vague language.

Severity scale 0-10 (higher = worse):
- fatigue: how tired/exhausted (0=none, 10=cannot function)
- joint_pain: aggregate joint pain
- brain_fog_severity: cognitive impairment (0=clear-headed, 10=cannot think)
- gi_distress: nausea, abdominal pain, bowel issues
- skin_severity: rash, itching, lesions
- sleep_severity: poor sleep quality last night (0=excellent sleep, 10=no sleep)
- energy_severity: low energy (0=normal energy, 10=no energy)
- wellness_severity: overall feeling unwell (0=feel great, 10=feel terrible)

Mood (use null if not assessable from text):
- phq8_score: 0-24 estimate of depression severity (only if patient describes depression-related symptoms)
- gad7_score: 0-21 estimate of anxiety severity (only if patient describes anxiety)

Other:
- emotional_valence: -1 (very negative tone) to +1 (very positive tone), null if neutral or unclear
- new_symptom_mentions: list of any specific symptoms mentioned that aren't in the 8 above (e.g., "Raynaud's", "mouth ulcers", "hair loss")
- explicit_flare: true ONLY if patient explicitly says they are having a flare

Respond with ONLY valid JSON matching this schema (no prose, no markdown):
{
  "fatigue": null or number,
  "joint_pain": null or number,
  "brain_fog_severity": null or number,
  "gi_distress": null or number,
  "skin_severity": null or number,
  "sleep_severity": null or number,
  "energy_severity": null or number,
  "wellness_severity": null or number,
  "phq8_score": null or number,
  "gad7_score": null or number,
  "emotional_valence": null or number,
  "new_symptom_mentions": [list of strings],
  "explicit_flare": true or false
}

Patient transcript:
\"\"\"
{TRANSCRIPT}
\"\"\""""


class VoiceTranscriptSource:
    """Extract structured symptoms from voice transcript via Claude Haiku.

    Args:
        api_key: Override API key. If None, reads ANTHROPIC_API_KEY.
        cache_dir: Override cache directory.
        model: Override model ID. Defaults to DEFAULT_EXTRACTION_MODEL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir or Path("./artifacts/agent5/nlp_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is None:
            self._client = get_anthropic_client(api_key=self.api_key)
        return self._client

    def extract(
        self,
        transcript: str,
        patient_id: str,
        target_date: str,
    ) -> FetchedSymptoms:
        """Extract structured symptoms from a transcript.

        Returns a FetchedSymptoms with errors populated if extraction fails;
        does not raise.
        """
        if not transcript or not transcript.strip():
            return FetchedSymptoms(errors=["VoiceTranscript: empty input"])

        text_hash = hashlib.md5(transcript.encode()).hexdigest()[:12]
        cache_path = (
            self.cache_dir / f"voice_{patient_id}_{target_date}_{text_hash}.pkl"
        )
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        try:
            client = self._get_client()
        except (RuntimeError, ImportError) as e:
            return FetchedSymptoms(errors=[f"VoiceTranscript setup: {e}"])

        prompt = _EXTRACTION_PROMPT.replace("{TRANSCRIPT}", transcript)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
        except Exception as e:
            return FetchedSymptoms(
                errors=[f"VoiceTranscript API: {type(e).__name__}: {e}"]
            )

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[-1].startswith("```"):
                raw_text = "\n".join(lines[1:-1])
            else:
                raw_text = "\n".join(lines[1:])

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            return FetchedSymptoms(
                errors=[
                    f"VoiceTranscript JSON parse: {e}",
                    f"raw: {raw_text[:200]}",
                ]
            )

        result = FetchedSymptoms(
            fatigue=data.get("fatigue"),
            joint_pain=data.get("joint_pain"),
            brain_fog_severity=data.get("brain_fog_severity"),
            gi_distress=data.get("gi_distress"),
            skin_severity=data.get("skin_severity"),
            sleep_severity=data.get("sleep_severity"),
            energy_severity=data.get("energy_severity"),
            wellness_severity=data.get("wellness_severity"),
            phq8_score=data.get("phq8_score"),
            gad7_score=data.get("gad7_score"),
            emotional_valence=data.get("emotional_valence"),
            new_symptom_mentions=data.get("new_symptom_mentions", []),
            explicit_flare=data.get("explicit_flare", False),
        )

        for feat in ALL_FEATURES:
            if getattr(result, feat) is not None:
                result.confidence[feat] = "voice_extracted"
                result.sources[feat] = "claude-haiku"

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            result.errors.append(f"VoiceTranscript cache write: {e}")

        return result
