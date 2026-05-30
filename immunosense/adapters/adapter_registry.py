"""AdapterRegistry — the Conductor's directory of agent adapters.

The Conductor doesn't know the five agents directly. It knows the registry:
"here is a UserBucket; for each agent that has data, hand me the adapter
that knows how to run it." The registry maps agent_id -> wrapped adapter and
is the single place adapters are constructed.

Typical wiring (done by the caller / app / test harness):

    registry = AdapterRegistry()
    registry.register(BiomarkerAdapter(biomarker_agent))
    registry.register(SymptomsMoodAdapter(symptoms_agent))
    ...

Or, for the common case where you have all five live agents:

    registry = AdapterRegistry.from_agents([
        biomarker_agent, dietary_agent, environment_agent,
        wearable_agent, symptoms_agent,
    ])
"""

from __future__ import annotations

from typing import Iterable, Optional

from immunosense.adapters.base import BaseAgentAdapter
from immunosense.adapters.biomarker_adapter import BiomarkerAdapter
from immunosense.adapters.dietary_adapter import DietaryAdapter
from immunosense.adapters.environment_adapter import EnvironmentAdapter
from immunosense.adapters.symptoms_mood_adapter import SymptomsMoodAdapter
from immunosense.adapters.wearable_adapter import WearableAdapter
from immunosense.agents.base import BaseAgent

# Maps agent_id -> the adapter class that wraps it.
ADAPTER_FOR_AGENT = {
    "agent1_biomarker": BiomarkerAdapter,
    "agent2_dietary": DietaryAdapter,
    "agent3_environment": EnvironmentAdapter,
    "agent4_wearable": WearableAdapter,
    "agent5_symptoms_mood": SymptomsMoodAdapter,
}


class AdapterRegistry:
    """Holds the set of adapters the Conductor can dispatch to."""

    def __init__(self):
        self._adapters: dict[str, BaseAgentAdapter] = {}

    def register(self, adapter: BaseAgentAdapter) -> None:
        """Register one adapter instance, keyed by its agent_id."""
        self._adapters[adapter.agent_id] = adapter

    def get(self, agent_id: str) -> Optional[BaseAgentAdapter]:
        return self._adapters.get(agent_id)

    def has(self, agent_id: str) -> bool:
        return agent_id in self._adapters

    @property
    def agent_ids(self) -> list:
        """Registered agent_ids, sorted."""
        return sorted(self._adapters.keys())

    def __len__(self) -> int:
        return len(self._adapters)

    # ------------------------------------------------------------------ #
    # Convenience construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_agents(cls, agents: Iterable[BaseAgent]) -> "AdapterRegistry":
        """Build a registry by wrapping each live agent in its adapter.

        Looks up the right adapter class by the agent's agent_id. Raises if
        an agent has no known adapter (a sign the registry map is stale).
        """
        registry = cls()
        for agent in agents:
            adapter_cls = ADAPTER_FOR_AGENT.get(agent.agent_id)
            if adapter_cls is None:
                raise ValueError(
                    f"No adapter registered for agent_id={agent.agent_id!r}. "
                    f"Known: {sorted(ADAPTER_FOR_AGENT)}"
                )
            registry.register(adapter_cls(agent))
        return registry
