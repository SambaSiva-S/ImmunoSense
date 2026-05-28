"""Agent adapters — thin translation + error isolation between Conductor and agents."""

from immunosense.adapters.adapter_registry import (
    ADAPTER_FOR_AGENT,
    AdapterRegistry,
)
from immunosense.adapters.base import (
    AdapterResult,
    AgentAdapter,
    BaseAgentAdapter,
)
from immunosense.adapters.biomarker_adapter import BiomarkerAdapter
from immunosense.adapters.dietary_adapter import DietaryAdapter
from immunosense.adapters.environment_adapter import EnvironmentAdapter
from immunosense.adapters.symptoms_mood_adapter import SymptomsMoodAdapter
from immunosense.adapters.wearable_adapter import WearableAdapter

__all__ = [
    "AgentAdapter",
    "BaseAgentAdapter",
    "AdapterResult",
    "AdapterRegistry",
    "ADAPTER_FOR_AGENT",
    "BiomarkerAdapter",
    "DietaryAdapter",
    "EnvironmentAdapter",
    "WearableAdapter",
    "SymptomsMoodAdapter",
]
