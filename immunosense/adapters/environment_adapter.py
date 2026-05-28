"""Environment (Agent 3) adapter.

process() input contract (from audit):
    {'daily_summary': DailyEnvironmentSummary,
     'flare_event': {'date': str, 'severity': float} (optional)}

The caller builds the DailyEnvironmentSummary via the environment Layer 2
pipeline (Option B). Optional flare event via AgentData.extras['flare_event'].
"""

from __future__ import annotations

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.events.bucket import AgentData


class EnvironmentAdapter(BaseAgentAdapter):
    agent_id = "agent3_environment"

    def build_input_data(self, agent_data: AgentData) -> dict:
        summary = agent_data.domain_object
        if summary is None:
            raise ValueError(
                "EnvironmentAdapter expects a DailyEnvironmentSummary domain_object"
            )
        input_data = {"daily_summary": summary}
        flare_event = agent_data.extras.get("flare_event")
        if flare_event is not None:
            input_data["flare_event"] = flare_event
        return input_data
