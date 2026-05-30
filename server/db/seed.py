"""Seed the reference schema from the library's canonical Python definitions.

The calibration LR table and corroboration patterns live in the library
(immunosense.conductor.calibration / fusion). This seeder copies them into the
reference schema so the DB holds a queryable, versioned copy — which the
Auto-Research loop (Sprint 9) will later read and write new versions of.

Idempotent: re-running updates existing rows rather than duplicating.
The knowledge table is intentionally left empty (NullKB; real KB later).
"""

from __future__ import annotations

from immunosense.conductor.calibration.likelihood_ratios import load_calibration
from immunosense.conductor.fusion.corroboration import Corroboration
from server.db.models import CalibrationLR, CorroborationPatternRow


def seed_reference_data(session_factory) -> dict:
    """Populate reference.calibration_lr and reference.corroboration_patterns.

    Returns a summary dict of how many rows were written.
    """
    cal = load_calibration()
    corr = Corroboration()

    n_lr = 0
    n_patterns = 0

    with session_factory() as session:
        # Likelihood ratios
        for agent_id in cal.agent_ids:
            lr = cal.get(agent_id)
            existing = session.get(CalibrationLR, {"version": cal.version, "agent_id": agent_id})
            if existing:
                existing.lr_positive = lr.lr_positive
                existing.lr_negative = lr.lr_negative
                existing.signal_threshold = lr.signal_threshold
                existing.low_threshold = lr.low_threshold
                existing.source = lr.source
            else:
                session.add(
                    CalibrationLR(
                        version=cal.version,
                        agent_id=agent_id,
                        lr_positive=lr.lr_positive,
                        lr_negative=lr.lr_negative,
                        signal_threshold=lr.signal_threshold,
                        low_threshold=lr.low_threshold,
                        source=lr.source,
                    )
                )
            n_lr += 1

        # Corroboration patterns
        for pat in corr.patterns:
            existing = session.get(CorroborationPatternRow, pat.name)
            if existing:
                existing.label = pat.label
                existing.description = pat.description
                existing.required_agents = list(pat.required_agents)
                existing.optional_agents = list(pat.optional_agents)
                existing.source = pat.source
            else:
                session.add(
                    CorroborationPatternRow(
                        pattern_id=pat.name,
                        label=pat.label,
                        description=pat.description,
                        required_agents=list(pat.required_agents),
                        optional_agents=list(pat.optional_agents),
                        source=pat.source,
                    )
                )
            n_patterns += 1

        session.commit()

    return {
        "calibration_version": cal.version,
        "lr_rows": n_lr,
        "pattern_rows": n_patterns,
    }
