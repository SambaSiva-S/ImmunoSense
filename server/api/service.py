"""Service layer — the bridge between stored rows and the Conductor.

Given a user + a time window (bucket), this:
  1. reads the raw logs from the DB for that bucket's date
  2. runs the builders to produce agent domain objects
  3. assembles a UserBucket
  4. runs the Conductor (which persists Layer A events + the report)
  5. returns the ConductorReport

This is where the builders, data layer, and Conductor compose. The API routes
are thin wrappers over this.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from immunosense.adapters import (
    AdapterRegistry,
    BiomarkerAdapter,
    DietaryAdapter,
    EnvironmentAdapter,
    SymptomsMoodAdapter,
)
from immunosense.agents.biomarker.agent import BiomarkerAgent
from immunosense.agents.dietary.agent import DietaryAgent
from immunosense.agents.environment.agent import EnvironmentAgent
from immunosense.agents.symptoms_mood.agent import SymptomsMoodAgent
from immunosense.conductor import Conductor
from immunosense.events import AgentData, BucketBuilder, UserBucket
from server.api.config import Settings
from server.api.tracelog import get_trace_id, log
from server.builders import (
    DietaryPipeline,
    build_biomarker_input,
    build_symptom_summary,
)
from server.db.event_store import PostgresEventLog
from server.db.models import (
    BiomarkerReading,
    DietaryLog,
    FlareButtonEvent,
    Profile,
    SymptomLog,
)


class EvaluationService:
    """Builds buckets from stored rows and runs the Conductor for a user."""

    def __init__(self, session_factory: sessionmaker, settings: Settings):
        self.session_factory = session_factory
        self.settings = settings
        self._dietary_pipeline: Optional[DietaryPipeline] = None
        # Build the dietary pipeline lazily, once, if caches are configured.
        if settings.dietary_density_cache and settings.dietary_food_index_cache:
            try:
                self._dietary_pipeline = DietaryPipeline(
                    settings.dietary_density_cache,
                    settings.dietary_food_index_cache,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(f"Dietary cache load failed; dietary disabled: {exc}")

    def _make_conductor(self, disease: str) -> Conductor:
        # Only the 3 viable Phase 1 agents are registered. Wearable/environment
        # are dormant (no Phase 1 data source); the Conductor handles their
        # absence gracefully.
        registry = AdapterRegistry()
        registry.register(SymptomsMoodAdapter(SymptomsMoodAgent()))
        registry.register(BiomarkerAdapter(BiomarkerAgent()))
        registry.register(DietaryAdapter(DietaryAgent()))
        # Environment (Agent 3): fed from the user's home location when set.
        # CompositeSource uses real APIs when keys are configured, mock otherwise.
        registry.register(EnvironmentAdapter(EnvironmentAgent()))

        tfm = None
        if self.settings.use_claude_tfm:
            from immunosense.tfm import ClaudeTFM
            tfm = ClaudeTFM()

        log_store = PostgresEventLog(self.session_factory)
        kwargs = dict(registry=registry, event_log=log_store, disease=disease)
        if tfm is not None:
            kwargs["tfm"] = tfm
        return Conductor(**kwargs)

    def _disease_for(self, user_id: str) -> str:
        with self.session_factory() as s:
            prof = s.execute(
                select(Profile).where(Profile.user_id == user_id)
            ).scalar_one_or_none()
            if prof and prof.disease:
                return prof.disease
        return self.settings.default_disease

    def _demographics(self, user_id: str):
        with self.session_factory() as s:
            prof = s.execute(
                select(Profile).where(Profile.user_id == user_id)
            ).scalar_one_or_none()
        if prof is None:
            log.warning(f"no profile for {user_id}; agents use default demographics")
            return None, None, None
        age = _age_from_dob(getattr(prof, "date_of_birth", None))
        sex = getattr(prof, "sex", None)
        bmi = _bmi(getattr(prof, "height_cm", None), getattr(prof, "weight_kg", None))
        missing = [n for n, v in (("age", age), ("sex", sex), ("bmi", bmi)) if v is None]
        if missing:
            log.warning(f"{user_id} missing demographics {missing}; agents use defaults for those")
        return age, sex, bmi

    def build_user_bucket(self, user_id: str, ts: datetime, date: str) -> UserBucket:
        """Read this user's rows for `date` and assemble a UserBucket."""
        disease = self._disease_for(user_id)
        bucket = BucketBuilder.bucket_for(user_id, ts)
        ub = UserBucket(bucket=bucket)

        with self.session_factory() as s:
            sym_rows = s.execute(
                select(SymptomLog).where(SymptomLog.user_id == user_id)
                .where(SymptomLog.bucket_id == bucket.bucket_id)
            ).scalars().all()
            bio_rows = s.execute(
                select(BiomarkerReading).where(BiomarkerReading.user_id == user_id)
            ).scalars().all()
            meal_rows = s.execute(
                select(DietaryLog).where(DietaryLog.user_id == user_id)
                .where(DietaryLog.bucket_id == bucket.bucket_id)
            ).scalars().all()
            flare_rows = s.execute(
                select(FlareButtonEvent).where(FlareButtonEvent.user_id == user_id)
                .where(FlareButtonEvent.bucket_id == bucket.bucket_id)
            ).scalars().all()

        # Symptoms
        if sym_rows:
            flare_pressed = len(flare_rows) > 0
            flare_sev = max((f.severity for f in flare_rows), default=None)
            summary = build_symptom_summary(
                sym_rows, date=date, user_id=user_id, disease=disease,
                flare_button_pressed=flare_pressed,
                explicit_flare_severity=flare_sev,
            )
            ub.add(AgentData("agent5_symptoms_mood", summary, produced_at=ts))

        # Biomarker (most recent reading)
        if bio_rows:
            latest = max(bio_rows, key=lambda r: r.measured_at)
            age, sex, bmi = self._demographics(user_id)
            demo = _Demographics(age=age, sex=sex, bmi=bmi)
            bio_input = build_biomarker_input(latest, demo)
            ub.add(AgentData("agent1_biomarker", bio_input, produced_at=ts))

        # Dietary (only if caches are configured)
        if meal_rows and self._dietary_pipeline is not None:
            age, sex, bmi = self._demographics(user_id)
            rollup = self._dietary_pipeline.build_rollup(
                meal_rows, date=date, age=age, sex=sex, bmi=bmi
            )
            ub.add(AgentData("agent2_dietary", rollup, produced_at=ts))
        elif meal_rows:
            log.info("Meals present but dietary caches not configured; skipping dietary.")

        # Environment (Agent 3): only if the user has set a home location.
        # Best-effort — a data-source failure must never break the check-in.
        try:
            with self.session_factory() as s:
                prof = s.execute(
                    select(Profile).where(Profile.user_id == user_id)
                ).scalar_one_or_none()
            home_lat = getattr(prof, "home_lat", None) if prof else None
            home_lng = getattr(prof, "home_lng", None) if prof else None
            if home_lat is not None and home_lng is not None:
                from server.builders.environment_builder import build_environment_summary
                env_summary = build_environment_summary(
                    lat=float(home_lat), lon=float(home_lng), target_date=date,
                    label=getattr(prof, "home_label", None),
                )
                ub.add(AgentData("agent3_environment", env_summary, produced_at=ts))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"environment agent skipped for {user_id}: {exc}")

        return ub

    def evaluate(self, user_id: str, ts: Optional[datetime] = None):
        """Evaluate the user's bucket for the given (or current) timestamp."""
        ts = ts or datetime.now(timezone.utc)
        date = ts.strftime("%Y-%m-%d")
        ub = self.build_user_bucket(user_id, ts, date)
        conductor = self._make_conductor(self._disease_for(user_id))
        report = conductor.evaluate_bucket(ub)
        self._persist_report(report)
        log.info(
            f"evaluated bucket {ub.bucket_id} for {user_id}: "
            f"prob={report.flare_probability} conf={report.confidence_level.value}"
        )
        return report

    @staticmethod
    def debug_view(report) -> dict:
        """Rich, builder-facing view of a ConductorReport — the agent internals
        the wellness response intentionally hides. DEV ONLY (never exposed in
        production responses). Surfaces what each agent contributed, quality,
        fusion, and the raw inference numbers, so the builder can verify the
        agents are reacting correctly.
        """
        agents = []
        for agent_id in sorted(report.agent_results.keys()):
            res = report.agent_results[agent_id]
            q = report.agent_quality.get(agent_id)
            out = getattr(res, "output", None)
            agents.append({
                "agent_id": agent_id,
                "ok": getattr(res, "ok", None),
                "error": getattr(res, "error", None),
                "latency_ms": getattr(res, "latency_ms", None),
                "vector_dim": getattr(out, "vector_dim", None) if out else None,
                "confidence": getattr(out, "confidence", None) if out else None,
                "alerts": list(getattr(out, "alerts", []) or []) if out else [],
                "quality": None if q is None else {
                    "raw_confidence": q.raw_confidence,
                    "freshness": q.freshness,
                    "quality": q.quality,
                    "reported": q.reported,
                    "ok": q.ok,
                },
            })
        return {
            "user_id": report.user_id,
            "bucket_id": report.bucket_id,
            "evaluated_at": report.evaluated_at.isoformat() if report.evaluated_at else None,
            "trace_id": report.trace_id,
            "reporting_agents": report.reporting_agents,
            "agents": agents,
            "confidence_level": report.confidence_level.value,
            "overall_quality": report.overall_quality,
            "flare_probability": report.flare_probability,
            "severity_composite": report.severity_composite,
            "severity_band": report.severity_band,
            "matched_patterns": [
                {"name": p.name, "label": p.label, "description": p.description}
                for p in report.matched_patterns
            ],
            "fusion_contributions": list(report.fusion_contributions or []),
            "embedding_concat_dim": report.embedding_concat_dim,
            "calibration_version": report.calibration_version,
            "tfm_ok": report.tfm_ok,
            "explanation": report.explanation,
            "warnings": list(report.warnings or []),
            "errors": list(report.errors or []),
        }

    def _persist_report(self, report) -> None:
        """Upsert the ConductorReport summary into health.bucket_reports.

        The Conductor writes Layer A events; the server owns the report TABLE
        (the library doesn't know about server schemas). Last-write-wins per
        (user, bucket).
        """
        from server.db.models import BucketReport

        patterns = [
            {"name": p.name, "label": p.label, "description": p.description}
            for p in report.matched_patterns
        ]
        decision = None
        if getattr(report, "decision", None) is not None:
            decision = {
                "raise_alert": report.decision.raise_alert,
                "call_tfm": report.decision.call_tfm,
            }
        with self.session_factory() as s:
            existing = s.get(BucketReport, {"user_id": report.user_id,
                                            "bucket_id": report.bucket_id})
            if existing:
                existing.evaluated_at = report.evaluated_at
                existing.confidence_level = report.confidence_level.value
                existing.overall_quality = report.overall_quality
                existing.flare_probability = report.flare_probability
                existing.severity_composite = report.severity_composite
                existing.severity_band = report.severity_band
                existing.matched_patterns = patterns
                existing.decision = decision
                existing.explanation = report.explanation
                existing.calibration_version = report.calibration_version
                existing.trace_id = get_trace_id()
            else:
                s.add(BucketReport(
                    user_id=report.user_id, bucket_id=report.bucket_id,
                    evaluated_at=report.evaluated_at,
                    confidence_level=report.confidence_level.value,
                    overall_quality=report.overall_quality,
                    flare_probability=report.flare_probability,
                    severity_composite=report.severity_composite,
                    severity_band=report.severity_band,
                    matched_patterns=patterns, decision=decision,
                    explanation=report.explanation,
                    calibration_version=report.calibration_version,
                    trace_id=get_trace_id(),
                ))
            s.commit()


class _Demographics:
    """Carries computed demographics to the biomarker builder, which reads
    .age / .sex / .bmi. Age is derived from date_of_birth and BMI from
    height/weight upstream; this just exposes the results in the shape the
    builder expects (it falls back to its own defaults for any None)."""
    def __init__(self, age=None, sex=None, bmi=None):
        self.age = age
        self.sex = sex
        self.bmi = bmi


def _age_from_dob(dob: Optional[str]) -> Optional[float]:
    """Compute age in years from a 'YYYY-MM-DD' date_of_birth string."""
    if not dob:
        return None
    try:
        from datetime import date
        y, m, d = (int(x) for x in str(dob)[:10].split("-"))
        today = date.today()
        years = today.year - y - ((today.month, today.day) < (m, d))
        return float(years) if 0 < years < 130 else None
    except (ValueError, TypeError):
        return None


def _bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    """BMI = kg / m^2. None unless both inputs are present and sane."""
    if not height_cm or not weight_kg or height_cm <= 0:
        return None
    m = height_cm / 100.0
    val = weight_kg / (m * m)
    return round(val, 1) if 8 < val < 100 else None
