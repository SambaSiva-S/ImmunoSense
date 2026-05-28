"""Dietary (Agent 2) adapter.

process() input contract (from audit):
    {'rollup': DailyRollup, 'flares': [(date, severity), ...] (optional)}

The caller builds the DailyRollup via the dietary Layer 2 pipeline (Option B).
Optional flare history may be passed through AgentData.extras['flares'].
"""

from __future__ import annotations

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.events.bucket import AgentData


class DietaryAdapter(BaseAgentAdapter):
    agent_id = "agent2_dietary"

    def build_input_data(self, agent_data: AgentData) -> dict:
        rollup = agent_data.domain_object
        if rollup is None:
            raise ValueError("DietaryAdapter expects a DailyRollup domain_object")
        input_data = {"rollup": rollup}
        flares = agent_data.extras.get("flares")
        if flares is not None:
            input_data["flares"] = flares
        return input_data
