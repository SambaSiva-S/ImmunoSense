"""The Thinking Machine (TFM) — swappable explanation layer (Challenge 2).

v1 default: ClaudeTFM. Tests: MockTFM. Future local-model swap: LocalLLMTFM.
All implement the ThinkingMachine protocol and share the grounded prompt in
tfm.base, so swapping model backends changes one construction line, nothing else.
"""

from immunosense.tfm.base import (
    ThinkingMachine,
    TFMRequest,
    TFMResponse,
    build_prompt,
    fallback_explanation,
)
from immunosense.tfm.claude_tfm import ClaudeTFM
from immunosense.tfm.local_llm_tfm import LocalLLMTFM
from immunosense.tfm.mock_tfm import MockTFM

__all__ = [
    "ThinkingMachine",
    "TFMRequest",
    "TFMResponse",
    "build_prompt",
    "fallback_explanation",
    "MockTFM",
    "ClaudeTFM",
    "LocalLLMTFM",
]
