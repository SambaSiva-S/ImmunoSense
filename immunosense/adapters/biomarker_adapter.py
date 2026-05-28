"""Biomarker (Agent 1) adapter.

process() input contract (from audit):
    {'demographics': {'age': float, 'sex': int, 'bmi': float},
     'reading': {biomarker values + trigger booleans}}

Under Option B the caller supplies a domain_object that is already this dict
(or carries the two parts). The adapter just routes it.
"""

from __future__ import annotations

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.events.bucket import AgentData


class BiomarkerAdapter(BaseAgentAdapter):
    agent_id = "agent1_biomarker"

    def build_input_data(self, agent_data: AgentData) -> dict:
        obj = agent_data.domain_object
        # Accept either a ready-made {demographics, reading} dict, or an
        # object exposing those as attributes.
        if isinstance(obj, dict) and "demographics" in obj and "reading" in obj:
            return {"demographics": obj["demographics"], "reading": obj["reading"]}
        demographics = getattr(obj, "demographics", None)
        reading = getattr(obj, "reading", None)
        if demographics is None or reading is None:
            raise ValueError(
                "BiomarkerAdapter expects domain_object with 'demographics' "
                "and 'reading' (dict keys or attributes)"
            )
        return {"demographics": demographics, "reading": reading}
