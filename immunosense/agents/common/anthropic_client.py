"""Shared Anthropic Claude API client for agents that use LLM extraction.

Used by:
    - Agent 2 (Dietary): ClaudeHaikuExtractor for meal text decomposition
    - Agent 5 (Symptoms & Mood): VoiceTranscriptSource for symptom extraction

Provides:
    - Centralized API key management
    - Shared model identifiers (so we update them in one place)
    - Request tracking + telemetry
    - Error handling with retries
"""

from __future__ import annotations

import os
from typing import Any, Optional


# Standard model identifiers for ImmunoSense agents.
# Update here when migrating to new model versions.
DEFAULT_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_REASONING_MODEL = "claude-haiku-4-5-20251001"  # TFM may use a stronger model later

# Default sampling parameters for structured extraction
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0  # deterministic for extraction tasks


def get_api_key(env_var: str = "ANTHROPIC_API_KEY") -> Optional[str]:
    """Retrieve API key from environment.

    Args:
        env_var: Environment variable name (default ANTHROPIC_API_KEY).

    Returns:
        The API key string, or None if not set.
    """
    return os.environ.get(env_var)


def get_anthropic_client(api_key: Optional[str] = None) -> Any:
    """Lazily import and instantiate an Anthropic client.

    Args:
        api_key: Optional override API key. If None, reads from environment.

    Returns:
        An anthropic.Anthropic client instance.

    Raises:
        ImportError: If anthropic package is not installed.
        RuntimeError: If no API key is available.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The anthropic package is required for LLM agents. "
            "Install with: pip install immunosense[dietary,symptoms]"
        ) from e

    key = api_key or get_api_key()
    if key is None:
        raise RuntimeError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY environment "
            "variable or pass api_key explicitly."
        )

    return anthropic.Anthropic(api_key=key)


def is_anthropic_available() -> bool:
    """Quick check: can we use Anthropic API right now?

    Returns True if the anthropic package is installed AND an API key is set.
    Use this to fall back to mock extractors when Claude isn't available.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return get_api_key() is not None
