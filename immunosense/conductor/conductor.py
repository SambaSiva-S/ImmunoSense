"""The Conductor — hub-and-spoke orchestrator for ImmunoSense.

Every 6 hours (or on a flare-button override) the Conductor evaluates one
PatientBucket. The Sprint 5 flow:

    1. Validate the bucket against the adapter registry.
    2. For each agent that has data, run it through its adapter
       (error-isolated) and emit an AGENT_OUTPUT or AGENT_ERROR event.
    3. Score each agent's quality (confidence x freshness vs cadence).
    4. Aggregate into a 4-level ConfidenceLevel (Challenge 7).
    5. (Sprint 6 stubs) fusion -> probability, corroboration -> patterns,
       risk engine -> severity composite, decision -> TFM/alert policy.
    6. Emit a BUCKET_EVAL event summarizing the report.
    7. Return a ConductorReport.

The bottom half (fusion/decision) is wired but no-op in Sprint 5; the report's
inference fields stay None/empty until Sprint 6 fills the stubs in.

Hub-and-spoke (Challenge 2): the Conductor talks only to adapters + Layer A,
never to agents directly, and agents never talk to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from immunosense.adapters.adapter_registry import AdapterRegistry
from immunosense.adapters.base import AdapterResult
from immunosense.conductor.decision.decision_maker import Decision, DecisionMaker
from immunosense.conductor.fusion.corroboration import Corroboration
from immunosense.conductor.fusion.risk_engine import RiskEngine
from immunosense.conductor.fusion.statistical_fusion import StatisticalFusion
from immunosense.conductor.quality.confidence import (
    ConfidenceAggregator,
    ConfidenceResult,
)
from immunosense.conductor.quality.scorer import AgentQuality, QualityScorer
from immunosense.conductor.utils.trace import TraceContext
from immunosense.conductor.utils.validation import validate_patient_bucket
from immunosense.events.bucket import PatientBucket
from immunosense.events.event_log import EventLog
from immunosense.events.types import ConfidenceLevel, Event, EventType
from immunosense.inference.patient_day_embedding import build_patient_day_embedding
from immunosense.knowledge.base import KnowledgeBase, NullKB
from immunosense.tfm.base import TFMRequest, ThinkingMachine
from immunosense.tfm.mock_tfm import MockTFM


@dataclass
class ConductorReport:
    """The result of evaluating one bucket.

    Sprint 5 populates the top (observation + quality + confidence) half.
    The inference and curation halves stay at their defaults until Sprint 6
    and Sprint 7 respectively fill the stubs in.
    """

    patient_id: str
    bucket_id: str
    evaluated_at: datetime
    trace_id: str

    # --- Observations (Sprint 5) ---
    agent_results: dict = field(default_factory=dict)   # agent_id -> AdapterResult
    agent_quality: dict = field(default_factory=dict)   # agent_id -> AgentQuality

    # --- Confidence (Sprint 5, Challenge 7) ---
    confidence_level: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    overall_quality: float = 0.0

    # --- Inference (Sprint 6) ---
    flare_probability: Optional[float] = None
    matched_patterns: list = field(default_factory=list)
    severity_composite: Optional[float] = None
    severity_band: Optional[str] = None
    decision: Optional[Decision] = None
    explanation: Optional[str] = None          # TFM narrative (None if not called)
    tfm_ok: Optional[bool] = None              # True/False if TFM called, else None
    fusion_contributions: list = field(default_factory=list)  # per-agent audit
    calibration_version: Optional[str] = None
    embedding_concat_dim: Optional[int] = None  # JEPA envelope concat length

    # --- Episode curation (Sprint 7) ---
    is_memorable: bool = False
    episode_id: Optional[str] = None

    # --- Audit ---
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    source_event_ids: list = field(default_factory=list)

    @property
    def reporting_agents(self) -> list:
        return sorted(self.agent_results.keys())

    def summary(self) -> dict:
        """A compact, JSON-serializable summary for the BUCKET_EVAL event."""
        return {
            "patient_id": self.patient_id,
            "bucket_id": self.bucket_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "confidence_level": self.confidence_level.value,
            "overall_quality": round(self.overall_quality, 4),
            "reporting_agents": self.reporting_agents,
            "agent_quality": {
                aid: round(q.quality, 4) for aid, q in self.agent_quality.items()
            },
            "flare_probability": self.flare_probability,
            "matched_patterns": [
                p.name if hasattr(p, "name") else str(p) for p in self.matched_patterns
            ],
            "severity_composite": self.severity_composite,
            "severity_band": self.severity_band,
            "raised_alert": self.decision.raise_alert if self.decision else False,
            "tfm_called": self.decision.call_tfm if self.decision else False,
            "calibration_version": self.calibration_version,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
        }


class Conductor:
    """Hub-and-spoke orchestrator. Synchronous, per-bucket (Challenge 9)."""

    def __init__(
        self,
        registry: AdapterRegistry,
        event_log: EventLog,
        scorer: Optional[QualityScorer] = None,
        aggregator: Optional[ConfidenceAggregator] = None,
        fusion: Optional[StatisticalFusion] = None,
        corroboration: Optional[Corroboration] = None,
        risk_engine: Optional[RiskEngine] = None,
        decision_maker: Optional[DecisionMaker] = None,
        tfm: Optional[ThinkingMachine] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        disease: str = "autoimmune condition",
    ):
        self.registry = registry
        self.event_log = event_log
        self.scorer = scorer or QualityScorer()
        self.aggregator = aggregator or ConfidenceAggregator()
        # Sprint 6 inference components (now real implementations).
        self.fusion = fusion or StatisticalFusion()
        self.corroboration = corroboration or Corroboration()
        self.risk_engine = risk_engine or RiskEngine()
        self.decision_maker = decision_maker or DecisionMaker()
        # TFM defaults to the deterministic MockTFM so the Conductor is usable
        # (and testable) with no API key. Swap in ClaudeTFM / a local model in
        # production by passing tfm=ClaudeTFM().
        self.tfm = tfm or MockTFM()
        self.knowledge_base = knowledge_base or NullKB()
        self.disease = disease

    # ------------------------------------------------------------------ #
    # Main entrypoint
    # ------------------------------------------------------------------ #
    def evaluate_bucket(self, patient_bucket: PatientBucket) -> ConductorReport:
        """Evaluate one bucket end-to-end. Synchronous; returns when done."""
        ctx = TraceContext.for_bucket(patient_bucket.bucket_id)
        bucket = patient_bucket.bucket
        report = ConductorReport(
            patient_id=patient_bucket.patient_id,
            bucket_id=patient_bucket.bucket_id,
            evaluated_at=datetime.now(tz=bucket.end.tzinfo),
            trace_id=ctx.trace_id,
        )

        # 1. Validate
        report.warnings.extend(
            validate_patient_bucket(patient_bucket, self.registry)
        )

        # 2-3. Run each reporting agent through its adapter; score quality.
        qualities: list = []
        for agent_id in patient_bucket.reporting_agents:
            adapter = self.registry.get(agent_id)
            if adapter is None:
                # Already warned in validation; skip.
                continue

            agent_data = patient_bucket.get(agent_id)
            result: AdapterResult = adapter.run(
                agent_data, bucket_end=bucket.end, trace_id=ctx.trace_id
            )
            report.agent_results[agent_id] = result

            # Emit Layer A event (AGENT_OUTPUT or AGENT_ERROR).
            ev = self._event_for_result(patient_bucket, result)
            self.event_log.append(ev)
            report.source_event_ids.append(ev.event_id)
            if not result.ok:
                report.errors.append(f"{agent_id}: {result.error}")

            poll = getattr(adapter.agent, "poll_frequency", "daily")
            quality = self.scorer.score(result, agent_data, poll, bucket.end)
            report.agent_quality[agent_id] = quality
            qualities.append(quality)

        # Account for registered agents that did NOT report this bucket.
        for agent_id in self.registry.agent_ids:
            if agent_id not in report.agent_quality:
                absent = self.scorer.absent(agent_id)
                report.agent_quality[agent_id] = absent
                qualities.append(absent)

        # 4. Aggregate confidence (Challenge 7).
        conf: ConfidenceResult = self.aggregator.aggregate(qualities)
        report.confidence_level = conf.level
        report.overall_quality = conf.overall_quality

        # 5. Inference (Sprint 6 — real implementations).
        outputs = {aid: r.output for aid, r in report.agent_results.items() if r.ok}

        # Phase 1: Bayesian flare probability (the math truth).
        fusion_result = self.fusion.fuse(outputs, conf)
        report.flare_probability = fusion_result.flare_probability
        report.fusion_contributions = [
            {
                "agent_id": c.agent_id,
                "signal_strength": c.signal_strength,
                "direction": c.direction,
                "raw_lr": c.raw_lr,
                "quality": c.quality,
            }
            for c in fusion_result.contributions
        ]
        report.calibration_version = fusion_result.calibration_version

        # Phase 2: semantic corroboration patterns (no math feedback).
        report.matched_patterns = self.corroboration.match(outputs)

        # Phase 4: severity composite for UI/decisions (consumes Phase 1).
        risk_result = self.risk_engine.compute(
            report.flare_probability, conf, outputs
        )
        report.severity_composite = risk_result.severity_composite
        report.severity_band = risk_result.band

        # Decision policy: when to call TFM / raise alert.
        report.decision = self.decision_maker.decide(
            flare_probability=report.flare_probability,
            severity_composite=report.severity_composite,
            matched_patterns=report.matched_patterns,
            confidence_result=conf,
            flare_button=patient_bucket.flare_button,
            severity_band=report.severity_band,
        )

        # JEPA envelope (Challenge 5): assemble the combined daily embedding.
        pde = build_patient_day_embedding(
            patient_bucket.patient_id, patient_bucket.bucket_id, outputs
        )
        report.embedding_concat_dim = len(pde.to_concat())

        # TFM: only call when the decision policy says it's warranted.
        if report.decision.call_tfm:
            kb_context = [
                e.text
                for e in self.knowledge_base.query(
                    disease=self.disease,
                    tags=[c["agent_id"] for c in report.fusion_contributions],
                )
            ]
            tfm_request = TFMRequest(
                patient_id=patient_bucket.patient_id,
                bucket_id=patient_bucket.bucket_id,
                disease=self.disease,
                flare_probability=report.flare_probability,
                confidence_level=report.confidence_level.value,
                severity_composite=report.severity_composite,
                severity_band=report.severity_band,
                matched_patterns=[
                    {"name": p.name, "label": p.label, "description": p.description}
                    for p in report.matched_patterns
                ],
                agent_signals=report.fusion_contributions,
                kb_context=kb_context,
                audience=report.decision.audience,
            )
            tfm_request.trace_id = ctx.trace_id
            tfm_response = self.tfm.explain(tfm_request)
            report.explanation = tfm_response.explanation
            report.tfm_ok = tfm_response.ok

        # 6. Emit BUCKET_EVAL summary event.
        eval_ev = Event.create(
            patient_id=patient_bucket.patient_id,
            bucket_id=patient_bucket.bucket_id,
            event_type=EventType.BUCKET_EVAL,
            payload=report.summary(),
            quality=report.overall_quality,
            trace_id=ctx.trace_id,
        )
        self.event_log.append(eval_ev)
        report.source_event_ids.append(eval_ev.event_id)

        return report

    # ------------------------------------------------------------------ #
    # Critical-event override (Challenge 9)
    # ------------------------------------------------------------------ #
    def on_flare_button(
        self, patient_bucket: PatientBucket, severity: float
    ) -> ConductorReport:
        """Handle a flare-button press: log the event, then re-evaluate now.

        The flare button bypasses the 6h schedule. We record a FLARE_BUTTON
        event in Layer A first (so the override is auditable), then run a full
        bucket evaluation immediately.
        """
        ctx = TraceContext.for_bucket(patient_bucket.bucket_id)
        fb_event = Event.create(
            patient_id=patient_bucket.patient_id,
            bucket_id=patient_bucket.bucket_id,
            event_type=EventType.FLARE_BUTTON,
            payload={"severity": float(severity)},
            quality=1.0,
            trace_id=ctx.trace_id,
        )
        self.event_log.append(fb_event)

        # Make sure the bucket carries the flare button signal for downstream.
        patient_bucket.flare_button = float(severity)
        report = self.evaluate_bucket(patient_bucket)
        report.source_event_ids.insert(0, fb_event.event_id)
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _event_for_result(
        self, patient_bucket: PatientBucket, result: AdapterResult
    ) -> Event:
        """Build the Layer A event for one agent result.

        Serializes a SUMMARY of the AgentOutput — never the raw numpy vector.
        Layer A stays JSON-clean: we store the vector as a plain list and the
        structured alerts, which is enough for replay and audit.
        """
        out = result.output
        if result.ok:
            payload = {
                "vector": _vector_to_list(out.vector),
                "vector_dim": out.vector_dim,
                "alerts": out.alerts,
                "confidence": out.confidence,
                "data_keys": sorted(out.data.keys()) if isinstance(out.data, dict) else [],
                "latency_ms": round(result.latency_ms, 2),
            }
            etype = EventType.AGENT_OUTPUT
        else:
            payload = {
                "error": result.error,
                "latency_ms": round(result.latency_ms, 2),
            }
            etype = EventType.AGENT_ERROR

        return Event.create(
            patient_id=patient_bucket.patient_id,
            bucket_id=patient_bucket.bucket_id,
            event_type=etype,
            payload=payload,
            agent_id=result.agent_id,
            quality=out.confidence,
            trace_id=result.trace_id,
        )


def _vector_to_list(vector) -> list:
    """Convert a numpy vector to a JSON-safe list, tolerating None/NaN."""
    if vector is None:
        return []
    try:
        return [None if (v != v) else float(v) for v in vector]  # v!=v catches NaN
    except TypeError:
        return list(vector)
