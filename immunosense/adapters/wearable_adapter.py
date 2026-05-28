"""Wearable (Agent 4) adapter.

process() input contract (from audit):
    {'night_df': pd.DataFrame,     # minute-level signals
     'rr_intervals': list[float],  # RR intervals (ms)
     'night_idx': int,
     'is_flare': bool (optional)}

The caller assembles the raw night data (Option B). The domain_object is
expected to be a dict already in this shape, or an object exposing the
attributes; is_flare can also come via AgentData.extras.
"""

from __future__ import annotations

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.events.bucket import AgentData


class WearableAdapter(BaseAgentAdapter):
    agent_id = "agent4_wearable"

    _REQUIRED = ("night_df", "rr_intervals", "night_idx")

    def build_input_data(self, agent_data: AgentData) -> dict:
        obj = agent_data.domain_object
        if isinstance(obj, dict):
            data = dict(obj)  # shallow copy so we can augment safely
        else:
            data = {
                "night_df": getattr(obj, "night_df", None),
                "rr_intervals": getattr(obj, "rr_intervals", None),
                "night_idx": getattr(obj, "night_idx", None),
            }
            if hasattr(obj, "is_flare"):
                data["is_flare"] = obj.is_flare

        missing = [k for k in self._REQUIRED if data.get(k) is None]
        if missing:
            raise ValueError(
                f"WearableAdapter missing required keys: {missing}"
            )

        # Allow is_flare to be supplied via extras (synthetic-data testing).
        if "is_flare" not in data and "is_flare" in agent_data.extras:
            data["is_flare"] = agent_data.extras["is_flare"]
        return data
