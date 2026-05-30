"""Input validation for the Conductor.

Before evaluating a bucket the Conductor validates the UserBucket so
malformed input produces a clear error rather than a confusing failure deep
inside an adapter. Validation is intentionally lightweight — it checks
structure, not domain semantics (an adapter still error-isolates bad domain
objects).
"""

from __future__ import annotations

from immunosense.adapters.adapter_registry import AdapterRegistry
from immunosense.events.bucket import UserBucket


class BucketValidationError(ValueError):
    """Raised when a UserBucket is structurally invalid."""


def validate_user_bucket(
    user_bucket: UserBucket,
    registry: AdapterRegistry,
) -> list:
    """Validate a UserBucket against a registry.

    Checks:
        - bucket is present and has a user_id
        - every agent_id in agent_data has a registered adapter
        - flare_button severity, if present, is in [0, 1]

    Returns a list of non-fatal WARNING strings (e.g. an agent with data but
    no registered adapter is skipped, not fatal). Raises
    BucketValidationError only for structural problems that make evaluation
    impossible.
    """
    warnings: list = []

    if user_bucket is None:
        raise BucketValidationError("user_bucket is None")
    if user_bucket.bucket is None:
        raise BucketValidationError("user_bucket.bucket is None")
    if not user_bucket.user_id:
        raise BucketValidationError("user_bucket has empty user_id")

    for agent_id in user_bucket.reporting_agents:
        if not registry.has(agent_id):
            warnings.append(
                f"agent_data contains {agent_id!r} but no adapter is "
                f"registered for it; this agent will be skipped"
            )

    fb = user_bucket.flare_button
    if fb is not None and not (0.0 <= float(fb) <= 1.0):
        raise BucketValidationError(
            f"flare_button severity {fb} out of range [0, 1]"
        )

    return warnings
