# ImmunoSense — Architecture Decisions

This document records the locked architectural decisions for ImmunoSense so they
live in the codebase, not only in design-session transcripts. It is the
canonical reference for why the system is shaped the way it is.

---

## The 10 Architectural Challenges

These were resolved with full reasoning during the architecture-design sessions
and are considered **locked**. Changing any of them requires a deliberate
revisit, not an incidental refactor.

| # | Challenge | Locked Solution |
|---|-----------|-----------------|
| 1 | Temporal granularities across agents | Three-layer architecture: **Layer A** immutable events (NDJSON), **Layer B** 6-hour buckets with quality scoring, **Layer C** inference objects. |
| 2 | Notebook → module + interface inconsistency | Surgical extraction to the `immunosense/` package; notebooks become demos; pytest; TFM as a swappable abstraction (ClaudeTFM v1); HIPAA-compatible decisions baked in. |
| 3 | Cross-agent inference | Three-phase: **Phase 1** Bayesian aggregation (math truth) + **Phase 2** corroboration patterns (semantic only, no math feedback) + **Phase 3** selective TFM + **Phase 4** severity composite. No double-counting. |
| 4 | Mixed validation maturity | pytest + custom reporting + SQLite history + dashboard, built alongside each component (~57h total over the project). |
| 5 | JEPA dimension mismatch | `JEPACompatible` protocol + `PatientDayEmbedding` envelope. Each agent keeps its native dim (Agent5=36, Agent4=29, Agent1=7, Agent2=10, Agent3=5). Model architecture deferred to v2. |
| 6 | *(folded into Challenge 3)* | — |
| 7 | Missing data handling | Quality-aware, **4 confidence levels** (insufficient / low / moderate / high). Hard threshold for "insufficient" suppresses probability display. |
| 8 | Storage / recall (MEM0) | SQLite + sentence-transformer (all-MiniLM-L6-v2, 384-dim) + versioned embeddings + hybrid temporal/semantic query. **Curation lives in the Conductor**, not in `mem0/`. |
| 9 | Real-time vs batch | Synchronous per-bucket (6h) + critical-event override (flare button) + weekly Auto-Research batch. No Kafka/Celery/async in v1. |
| 10 | Auto-Research feedback loops | Level 1 read-only weekly review (always) + Level 2 bounded per-patient calibration (after 90 days, gated) + Level 3 cross-patient flagging (deferred) + Level 4 clinical-reviewed defaults (deferred). Intervention-aware. |

### Challenge 10 — failure modes to architect against

1. **The confirmation loop** — a patient who acts on a high prediction prevents
   the flare, which then looks like a false positive and wrongly lowers
   confidence. Prediction and intervention don't separate cleanly in chronic
   disease.
2. **Overfitting to one patient** — per-patient calibration that bleeds into
   other patients via shared defaults.
3. **Silent drift** — bounded auto-adjustments accumulate over months until the
   math no longer matches established immunology, eroding clinician trust.

---

## Sprint → Challenge implementation roadmap

| Sprint | Status | Challenges | What it builds |
|--------|--------|-----------|----------------|
| 1+2 | done | C2, C4 (partial) | Foundation + Agents 3, 4, 5 extracted |
| 3 | done | C2, C4 | Agent 2 (Dietary) |
| 4 | done | C2, C5, C4 | Agent 1 (Biomarker) |
| **5** | **this sprint** | **C1, C7, C9** | **Conductor + Layer A + Adapters** |
| 6 | planned | C3, C5 | Fusion (Bayesian + corroboration + risk) + TFM + JEPA envelope |
| 7 | planned | C8 | MEM0 long-term memory |
| 8 | planned | C2 final | Agents 6, 7, 8 |
| 9 | planned | C10 Level 1 | Auto-Research weekly review |
| v2 | deferred | C5 model, C10 L2–4 | Real JEPA training; per-patient & cross-patient learning |

---

## Sprint 5 — Conductor design (locked)

### Layering — Option B

The **caller** (ingestion / app / test harness) builds each agent's domain
object using that agent's existing Layer 2 pipeline, and drops them into a
`PatientBucket`. Adapters stay **thin**: they translate that domain object into
the agent's `process()` input shape and isolate errors. Adapters never build
domain objects and never run ML.

```
ingestion → PatientBucket → Conductor.evaluate_bucket()
                                 ├─ AdapterRegistry → AgentAdapter.run()  (error-isolated)
                                 │      └─ emit AGENT_OUTPUT / AGENT_ERROR → Layer A
                                 ├─ QualityScorer   (confidence × freshness vs cadence)
                                 ├─ ConfidenceAggregator → 4-level ConfidenceLevel
                                 ├─ fusion/*        (Sprint 6 stubs → None/[])
                                 ├─ decision/*      (Sprint 6 stub  → no-op)
                                 └─ emit BUCKET_EVAL → Layer A
                            → ConductorReport
```

### Package structure

```
immunosense/
├── events/              # Layer A — shared infra (top level)
│   ├── types.py         # Event, EventType, ConfidenceLevel
│   ├── event_log.py     # NDJSON read/write
│   └── bucket.py        # TimeBucket, BucketBuilder, PatientBucket, freshness
├── adapters/            # agent wrappers (top level)
│   ├── base.py          # AgentAdapter Protocol, BaseAgentAdapter, AdapterResult
│   ├── adapter_registry.py
│   └── {biomarker,dietary,environment,wearable,symptoms_mood}_adapter.py
└── conductor/
    ├── conductor.py     # Conductor + ConductorReport
    ├── fusion/          # Sprint 6 STUBS: statistical_fusion, corroboration, risk_engine
    ├── quality/         # scorer (per-agent), confidence (4-level)
    ├── decision/        # Sprint 6 STUB: decision_maker
    └── utils/           # trace, validation
```

### Locked decisions

1. **Event schema** — frozen `Event(event_id, patient_id, timestamp, bucket_id,
   event_type, agent_id, payload, quality, trace_id, schema_version)`. NDJSON,
   one file per patient per day, append-only.
2. **Buckets** — UTC 6-hour grid (T0–T3). Patient-local bucketing is a v2 option.
3. **Reuse `AgentOutput`** from `agents.base`; adapters do **not** redefine it.
4. **Error isolation** — adapters never raise. On failure they return a
   zero-confidence `AgentOutput` and the Conductor emits an `AGENT_ERROR` event.
5. **Quality** = `AgentOutput.confidence` × `freshness_weight(poll_frequency,
   produced_at, reference)`. Freshness half-life is **relative to each agent's
   cadence** (1hr=6h, 6hr=18h, daily=36h, weekly=240h). This is the concrete
   realization of Challenge 7.
6. **Confidence aggregation** — 4-level rule (3-of-5 thresholds as the simplest
   defensible starting heuristic; refine later via Auto-Research).
7. **`ConductorReport`** — forward-compatible: Sprint 5 fills observation +
   quality + confidence; Sprint 6/7 fill inference + curation.
8. **Scheduler deferred** to Sprint 9; `evaluate_bucket()` is called directly.
   Flare button triggers immediate re-evaluation (Challenge 9 override).
9. **`events/` and `adapters/` are top-level** shared infrastructure; the
   Conductor consumes them. MEM0 (Sprint 7) and Auto-Research (Sprint 9) also
   consume Layer A, so it must not live under `conductor/`.

### Per-agent input contracts (from the Sprint 5 source audit)

| Agent | `process()` input | output_dim | poll_frequency |
|-------|-------------------|-----------|----------------|
| agent1_biomarker | `{demographics, reading}` | 7 | weekly |
| agent2_dietary | `{rollup, flares?}` | 10 | daily |
| agent3_environment | `{daily_summary, flare_event?}` | 5 | 6hr |
| agent4_wearable | `{night_df, rr_intervals, night_idx, is_flare?}` | 29 | 1hr |
| agent5_symptoms_mood | `{daily_summary}` | 36 | daily |

All five already return the same `AgentOutput` and already self-report
`confidence`, which is why adapters only translate **in**.
