"""Verify Sprint 6 installed correctly.

IMPORTANT: all imports are at MODULE LEVEL on purpose (same pattern as
verify_sprint5.py). Sprint 4's verifier wrapped imports in a function and
produced repeated false-negative failures. Do not move these inside main().

Run from the project root:
    venv\\Scripts\\python.exe verify_sprint6.py
"""

# --- Module-level imports: calibration + fusion ---
from immunosense.conductor.calibration import (
    AgentLR,
    CALIBRATION_VERSION,
    CalibrationTable,
    DEFAULT_BASELINE_FLARE_PRIOR,
    load_calibration,
)
from immunosense.conductor.fusion import (
    AgentContribution,
    Corroboration,
    CorroborationPattern,
    FusionResult,
    MatchedPattern,
    RiskEngine,
    RiskResult,
    StatisticalFusion,
    extract_signal_strength,
)

# --- Module-level imports: decision ---
from immunosense.conductor.decision import Decision, DecisionMaker

# --- Module-level imports: TFM ---
from immunosense.tfm import (
    ClaudeTFM,
    LocalLLMTFM,
    MockTFM,
    TFMRequest,
    TFMResponse,
    ThinkingMachine,
    build_prompt,
    fallback_explanation,
)

# --- Module-level imports: knowledge + inference ---
from immunosense.knowledge import KnowledgeBase, KnowledgeEntry, NullKB
from immunosense.inference import (
    EMBEDDING_LAYOUT_VERSION,
    JEPACompatible,
    PatientDayEmbedding,
    TOTAL_CONCAT_DIM,
    build_patient_day_embedding,
)

# --- Module-level imports: conductor integration ---
from immunosense.conductor import Conductor, ConductorReport
from immunosense.adapters import AdapterRegistry, SymptomsMoodAdapter, WearableAdapter
from immunosense.events import (
    AgentData,
    BucketBuilder,
    EventLog,
    EventType,
    PatientBucket,
)
from immunosense.agents.base import AgentOutput, BaseAgent

# --- Stdlib ---
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


# --- Fake agent for end-to-end smoke (no trained models needed) ---
class _ElevatedAgent(BaseAgent):
    def __init__(self, agent_id, dim, poll):
        super().__init__()
        self.agent_id = agent_id
        self.output_dim = dim
        self.poll_frequency = poll

    def process(self, input_data):
        return AgentOutput(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            data={"ok": True},
            vector=np.ones(self.output_dim) * 0.8,
            vector_dim=self.output_dim,
            alerts=[{"severity": "critical"}],
            confidence=0.9,
        )


@dataclass
class _Q:
    agent_id: str
    quality: float


@dataclass
class _Conf:
    level: object
    per_agent: list


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("Sprint 6 verification")
    print("=" * 50)

    print("\n1. Imports")
    check("calibration", load_calibration is not None)
    check("fusion (StatisticalFusion, Corroboration, RiskEngine)",
          all([StatisticalFusion, Corroboration, RiskEngine]))
    check("decision (DecisionMaker)", DecisionMaker is not None)
    check("tfm (Mock + Claude + Local + ThinkingMachine)",
          all([MockTFM, ClaudeTFM, LocalLLMTFM, ThinkingMachine]))
    check("knowledge (NullKB)", NullKB is not None)
    check("inference (PatientDayEmbedding)", PatientDayEmbedding is not None)

    print("\n2. Calibration table")
    cal = load_calibration()
    check(f"version {CALIBRATION_VERSION}", cal.version == CALIBRATION_VERSION)
    check(f"baseline prior = {DEFAULT_BASELINE_FLARE_PRIOR}",
          cal.baseline_prior == DEFAULT_BASELINE_FLARE_PRIOR)
    check("5 agent entries", len(cal.agent_ids) == 5)
    check("every entry has a source",
          all(cal.get(a).source for a in cal.agent_ids))

    print("\n3. Bayesian fusion math")
    fusion = StatisticalFusion()
    from immunosense.events.types import ConfidenceLevel
    # Gating
    r_gated = fusion.fuse({}, _Conf(ConfidenceLevel.INSUFFICIENT, []))
    check("INSUFFICIENT gates probability to None", r_gated.flare_probability is None)
    # Elevation
    out = AgentOutput(
        agent_id="agent5_symptoms_mood", timestamp=datetime.now(timezone.utc),
        data={}, vector=np.ones(36), vector_dim=36,
        alerts=[{"severity": "critical"}], confidence=0.9,
    )
    r = fusion.fuse(
        {"agent5_symptoms_mood": out},
        _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 1.0)]),
    )
    check("elevated signal raises prob above prior",
          r.flare_probability > DEFAULT_BASELINE_FLARE_PRIOR)
    # Quality tempering
    r_half = fusion.fuse(
        {"agent5_symptoms_mood": out},
        _Conf(ConfidenceLevel.HIGH, [_Q("agent5_symptoms_mood", 0.5)]),
    )
    check("half-quality update smaller than full-quality",
          r_half.flare_probability < r.flare_probability)

    print("\n4. Corroboration patterns")
    corr = Corroboration()
    check(f"{len(corr.patterns)} patterns (6-8 cross-disease)",
          6 <= len(corr.patterns) <= 8)
    out_w = AgentOutput(agent_id="agent4_wearable",
        timestamp=datetime.now(timezone.utc), data={},
        vector=np.ones(29), vector_dim=29,
        alerts=[{"severity": "critical"}], confidence=0.9)
    out_s = AgentOutput(agent_id="agent5_symptoms_mood",
        timestamp=datetime.now(timezone.utc), data={},
        vector=np.ones(36), vector_dim=36,
        alerts=[{"severity": "critical"}], confidence=0.9)
    matches = corr.match({"agent4_wearable": out_w, "agent5_symptoms_mood": out_s})
    check("wearable+symptoms -> autonomic_stress matches",
          any(m.name == "autonomic_stress" for m in matches))
    check("MatchedPattern has no probability field (no double-counting)",
          all(not hasattr(m, "probability") for m in matches))

    print("\n5. Risk engine + decision")
    risk = RiskEngine()
    rr = risk.compute(0.3, _Conf(ConfidenceLevel.HIGH, []),
                       {"agent5_symptoms_mood": out_s})
    check("composite in [0,1]", 0.0 <= rr.severity_composite <= 1.0)
    rr_gated = risk.compute(None, _Conf(ConfidenceLevel.INSUFFICIENT, []), {})
    check("risk gates to None when probability is None",
          rr_gated.severity_composite is None)
    dm = DecisionMaker()
    d_alert = dm.decide(0.7, 0.7, [], _Conf(ConfidenceLevel.HIGH, []), severity_band="high")
    check("high severity -> alert + tfm", d_alert.raise_alert and d_alert.call_tfm)
    d_button = dm.decide(None, None, [], _Conf(ConfidenceLevel.INSUFFICIENT, []),
                          flare_button=0.9)
    check("flare button honored under INSUFFICIENT", d_button.raise_alert)

    print("\n6. TFM layer (ThinkingMachine swappability)")
    check("MockTFM satisfies protocol", isinstance(MockTFM(), ThinkingMachine))
    check("ClaudeTFM satisfies protocol", isinstance(ClaudeTFM(), ThinkingMachine))
    check("LocalLLMTFM satisfies protocol", isinstance(LocalLLMTFM(), ThinkingMachine))
    # Mock determinism
    req = TFMRequest(patient_id="p", bucket_id="b", disease="SLE",
                     flare_probability=0.3, confidence_level="moderate",
                     severity_band="moderate")
    r1 = MockTFM().explain(req); r2 = MockTFM().explain(req)
    check("MockTFM deterministic", r1.explanation == r2.explanation)
    check("MockTFM output is safe (mentions clinician/diagnosis)",
          "clinician" in r1.explanation.lower() or "diagnosis" in r1.explanation.lower())
    # Claude fail-safe (no key)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    rc = ClaudeTFM(api_key=None).explain(req)
    check("ClaudeTFM no-key path degrades (does not raise)",
          rc.ok is False and rc.explanation)

    print("\n7. Knowledge seam (NullKB)")
    kb = NullKB()
    check("NullKB satisfies KnowledgeBase protocol", isinstance(kb, KnowledgeBase))
    check("NullKB returns empty grounding",
          kb.query(disease="SLE", tags=["hrv"]) == [])

    print("\n8. JEPA envelope (Challenge 5)")
    check(f"TOTAL_CONCAT_DIM = 87", TOTAL_CONCAT_DIM == 87)
    check(f"layout version {EMBEDDING_LAYOUT_VERSION}",
          EMBEDDING_LAYOUT_VERSION == "pde-v1")
    pde = PatientDayEmbedding(patient_id="p", bucket_id="b")
    check("empty -> 87-dim zero vector",
          len(pde.to_concat()) == 87 and np.allclose(pde.to_concat(), 0.0))
    pde.add("agent1_biomarker", np.ones(7) * 0.5)
    check("first 7 = biomarker block", np.allclose(pde.to_concat()[:7], 0.5))
    try:
        pde.add("agent1_biomarker", np.ones(5))  # wrong dim
        check("dim mismatch rejected", False)
    except ValueError:
        check("dim mismatch rejected on add", True)

    print("\n9. Conductor end-to-end integration")
    sym = _ElevatedAgent("agent5_symptoms_mood", 36, "daily")
    wear = _ElevatedAgent("agent4_wearable", 29, "1hr")
    registry = AdapterRegistry()
    registry.register(SymptomsMoodAdapter(sym))
    registry.register(WearableAdapter(wear))
    with tempfile.TemporaryDirectory() as tmp:
        log = EventLog(tmp)
        conductor = Conductor(registry=registry, event_log=log, disease="SLE")
        ts = datetime(2026, 5, 27, 14, 30, tzinfo=timezone.utc)
        bucket = BucketBuilder.bucket_for("patient001", ts)
        pb = PatientBucket(bucket=bucket)
        pb.add(AgentData("agent5_symptoms_mood", "summary", produced_at=ts))
        pb.add(AgentData("agent4_wearable",
                          {"night_df": "df", "rr_intervals": [800], "night_idx": 1},
                          produced_at=ts))
        report = conductor.evaluate_bucket(pb)
        check("flare_probability is not None (above baseline)",
              report.flare_probability is not None
              and report.flare_probability > DEFAULT_BASELINE_FLARE_PRIOR)
        check("severity_composite is in [0,1]",
              report.severity_composite is not None
              and 0.0 <= report.severity_composite <= 1.0)
        check("autonomic_stress pattern matched",
              any(p.name == "autonomic_stress" for p in report.matched_patterns))
        check("calibration_version recorded",
              report.calibration_version == "lr-v1")
        check("embedding_concat_dim = 87", report.embedding_concat_dim == 87)
        check("decision called TFM", report.decision.call_tfm)
        check("TFM produced explanation", report.explanation is not None)
        check("TFM ok=True", report.tfm_ok is True)
        # Layer A
        evts = log.read_bucket("patient001", bucket.bucket_id)
        check("BUCKET_EVAL event present",
              any(e.event_type == EventType.BUCKET_EVAL for e in evts))
        # All events share one trace
        trace_ids = {e.trace_id for e in evts}
        check("all events share one trace id", len(trace_ids) == 1)

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("Sprint 6 verified OK.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
