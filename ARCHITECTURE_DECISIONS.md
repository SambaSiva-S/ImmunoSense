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
| 5 | done | C1, C7, C9 | Conductor + Layer A + Adapters |
| **6** | **this sprint** | **C3, C5, C2** | **Fusion (Bayesian + corroboration + risk) + TFM + JEPA envelope** |
| 6.5 (new) | planned | — | **Data architecture + UI/App** (web portal + mobile, backend API, DB, auth — confirmed primary focus after Sprint 6) |
| 7 | planned | C8 | MEM0 long-term memory |
| 8 | planned | C2 final | Agents 6, 7 (molecular/genomics), 8 — gated by data architecture |
| 9 | planned | C10 Level 1 | Auto-Research weekly review |
| v2 | deferred | C5 model, C10 L2–4 | Real JEPA training; per-patient & cross-patient learning |
| later | deferred | — | Recommendation engine; Causal inference layer (require safety framework + longitudinal data) |

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

---

## Sprint 6 — Inference layer design (locked)

### What Sprint 6 built (Challenges 3, 5, 2)

```
immunosense/
├── conductor/
│   ├── calibration/             # Versioned LR table (lr-v1) — Q1 decision
│   │   └── likelihood_ratios.py
│   ├── fusion/
│   │   ├── statistical_fusion.py  # Bayesian probability (Phase 1 — math truth)
│   │   ├── corroboration.py       # 7 cross-disease patterns (Phase 2 — semantic only)
│   │   └── risk_engine.py         # Severity composite (Phase 4 — UI-facing)
│   └── decision/
│       └── decision_maker.py      # TFM/alert policy (Phase 3 + policy)
├── tfm/                         # Thinking Machine (Challenge 2)
│   ├── base.py                  # ThinkingMachine protocol + shared prompt
│   ├── mock_tfm.py              # Deterministic mock (tests use this)
│   ├── claude_tfm.py            # v1 default — fail-safe Anthropic API client
│   └── local_llm_tfm.py         # Scaffold for local Llama swap (Ollama/vLLM)
├── knowledge/                   # KB seam — NullKB now, real KB later
│   └── base.py
└── inference/
    └── patient_day_embedding.py # JEPACompatible + PatientDayEmbedding (Challenge 5)
```

### Locked decisions (Sprint 6)

1. **Likelihood ratios** are a **versioned calibration table** (`lr-v1`), seeded
   by reasoned defaults with literature direction noted on each entry. Every LR
   carries an explicit `source` field. Auto-Research (Sprint 9) writes new
   versions rather than mutating in place; patient history is interpreted against
   the LR version live when computed.

2. **Bayesian fusion** combines per-agent likelihood ratios in **log-odds space**
   with **quality tempering**: `effective_log_LR = quality * log(LR)`. A
   zero-quality agent contributes nothing; full quality contributes the full LR.
   **INSUFFICIENT confidence gates the probability to `None`** (Challenge 7).

3. **Corroboration is semantic-only** — 7 cross-disease patterns. `MatchedPattern`
   has no probability field; structural enforcement of the "no double-counting"
   rule (Challenge 3 design fix). Patterns are provisional (literature-informed
   or default-reasoned), each with a `source`.

4. **Risk engine consumes the probability**, blends with `acute_severity` (max
   signal across reporting agents), and **damps by confidence level**. Gates to
   `None` when probability is gated.

5. **Decision policy is separated from the math.** Default thresholds:
   `severity >= 0.6` or `probability >= 0.5` or flare button → alert. Pattern
   match or moderate/high band → call TFM. Insufficient confidence → no alert,
   TFM optional to explain the gap. **Flare button is honored even under
   insufficient confidence** (safety override).

6. **ClaudeTFM is the v1 default**, with `ThinkingMachine` abstraction making
   any other model (local Llama via Ollama/vLLM, mock) a drop-in swap. All
   backends share the same grounded, guardrailed prompt from `tfm.base`.

7. **NullKB now**; real KB later from autoimmune flare/disease-activity
   literature. The TFM handles empty `kb_context` gracefully.

8. **PatientDayEmbedding** envelope: each agent keeps its native dim; assembly
   produces a **stable 87-dim concatenation** in a fixed agent order with
   zero-blocks for absent agents and a presence mask. Layout version `pde-v1`.
   JEPA *model* architecture deferred to v2.

### TFM swappability (Challenge 2, concrete)

Three backends ship in `tfm/`, all conforming to the `ThinkingMachine` protocol:

- **MockTFM** — deterministic, no network/key/GPU. Tests use this. Also the
  Conductor default so the system runs out-of-the-box.
- **ClaudeTFM** — v1 production default. Lazy SDK import, lazy key check, **never
  raises into the Conductor** (any error → degraded `TFMResponse(ok=False)`
  carrying the safe fallback explanation).
- **LocalLLMTFM** — scaffold for a future Ollama / vLLM backend (e.g. Llama 3.1
  8B). Currently returns the safe fallback; documented swap target for
  HIPAA-friendly deployment.

Swap is one construction line: `Conductor(..., tfm=ClaudeTFM())`.

### Sprint 6 honest deferrals

These were intentionally deferred from Sprint 6, with the recorded reason:

- **Allen Institute / CELLxGENE molecular data (scRNA-seq, CellTypist models,
  clinical labs)** — evaluated during Sprint 6, found to be molecular reference
  data that does **not** fit the TFM's grounding slot (it's cell-level
  transcriptomics, not flare/disease-activity knowledge). It is recorded here as
  **candidate reference data for a future Agent 7 (molecular/genomics)** — to
  be built only after the data architecture and UI/App (Sprint 6.5) exist and
  the product decision on patient sequencing ingestion is made. License/use
  terms must be confirmed before any of this enters a clinical product.
- **Real KB content** for TFM grounding — deferred until literature sourcing
  is deliberate (autoimmune disease-activity, flare onset signals); NullKB
  bridges the gap.
- **Recommendation engine** and **Causal inference layer** — both require
  prerequisites we don't yet have: a clinical safety framework (recommendations)
  and longitudinal patient data (causal). Sequenced **after** the app exists
  and MEM0 / Auto-Research have accumulated history.

### Provisional content disclosure (Sprint 6)

The LR table values, the 7 corroboration patterns, and the LR/decision/risk
thresholds are **provisional starting values** — reasoned defaults, some
literature-informed where the direction is well-established. None should be
treated as clinically validated. They are tuning points the Auto-Research loop
is designed to calibrate (Sprint 9 read-only, deeper levels deferred to v2).
