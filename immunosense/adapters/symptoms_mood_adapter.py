"""Symptoms & Mood (Agent 5) adapter.

process() input contract (from audit):
    {'daily_summary': DailySymptomMoodSummary}

The caller builds the DailySymptomMoodSummary via the symptoms Layer 2
pipeline (Option B). This is the simplest adapter — single key passthrough.
"""

from __future__ import annotations

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.events.bucket import AgentData


class SymptomsMoodAdapter(BaseAgentAdapter):
    agent_id = "agent5_symptoms_mood"

    def build_input_data(self, agent_data: AgentData) -> dict:
        summary = agent_data.domain_object
        if summary is None:
            raise ValueError(
                "SymptomsMoodAdapter expects a DailySymptomMoodSummary domain_object"
            )
        return {"daily_summary": summary}
