"""Input validation for the Conductor.

Before evaluating a bucket the Conductor validates the PatientBucket so
malformed input produces a clear error rather than a confusing failure deep
inside an adapter. Validation is intentionally lightweight — it checks
structure, not domain semantics (an adapter still error-isolates bad domain
objects).
"""

from __future__ import annotations

from immunosense.adapters.adapter_registry import AdapterRegistry
from immunosense.events.bucket import PatientBucket


class BucketValidationError(ValueError):
    """Raised when a PatientBucket is structurally invalid."""


def validate_patient_bucket(
    patient_bucket: PatientBucket,
    registry: AdapterRegistry,
) -> list:
    """Validate a PatientBucket against a registry.

    Checks:
        - bucket is present and has a patient_id
        - every agent_id in agent_data has a registered adapter
        - flare_button severity, if present, is in [0, 1]

    Returns a list of non-fatal WARNING strings (e.g. an agent with data but
    no registered adapter is skipped, not fatal). Raises
    BucketValidationError only for structural problems that make evaluation
    impossible.
    """
    warnings: list = []

    if patient_bucket is None:
        raise BucketValidationError("patient_bucket is None")
    if patient_bucket.bucket is None:
        raise BucketValidationError("patient_bucket.bucket is None")
    if not patient_bucket.patient_id:
        raise BucketValidationError("patient_bucket has empty patient_id")

    for agent_id in patient_bucket.reporting_agents:
        if not registry.has(agent_id):
            warnings.append(
                f"agent_data contains {agent_id!r} but no adapter is "
                f"registered for it; this agent will be skipped"
            )

    fb = patient_bucket.flare_button
    if fb is not None and not (0.0 <= float(fb) <= 1.0):
        raise BucketValidationError(
            f"flare_button severity {fb} out of range [0, 1]"
        )

    return warnings
