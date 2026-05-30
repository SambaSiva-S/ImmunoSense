"""The ThinkingMachine (TFM) abstraction — the swappable explanation layer.

The TFM turns a finished quantitative evaluation (flare probability, matched
patterns, severity, agent signals) into a grounded NATURAL-LANGUAGE explanation
for the patient and/or clinician. It is the "thinking" layer that narrates the
math — it does NOT compute the math and must never override it.

WHY AN ABSTRACTION (Challenge 2, locked):
    The model behind the TFM is an implementation detail. v1 uses ClaudeTFM
    (Anthropic API). A local open-source model (Llama 3.1/3.3 via Ollama or
    vLLM) can be dropped in later with zero changes to the Conductor, because
    everything depends on this `ThinkingMachine` protocol, not on a concrete
    model. Tests run against MockTFM (deterministic, no network, no API key).

CONTRACT GUARANTEES every implementation must honor:
    1. The TFM receives a TFMRequest that already contains the computed
       probability/severity/patterns. It EXPLAINS them; it does not recompute.
    2. The TFM must respect the confidence floor: if the evaluation was gated
       (INSUFFICIENT), it is not asked to invent a confident narrative.
    3. The TFM must never present itself as medical advice or diagnosis. The
       prompt and post-processing enforce an explanatory, non-prescriptive
       stance. (A recommendation engine is a separate, later, gated component.)
    4. The TFM must be fail-safe: if the model is unreachable or errors, the
       implementation returns a degraded TFMResponse (ok=False) rather than
       raising, so a TFM outage never crashes a bucket evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class TFMRequest:
    """Everything the TFM needs to explain one evaluation.

    All quantitative fields are ALREADY COMPUTED upstream. The TFM consumes
    them read-only.

    Fields:
        user_id / bucket_id: identity for logging/trace.
        disease: the patient's condition (shapes the explanation context).
        flare_probability: Phase 1 posterior (None if gated).
        confidence_level: the Challenge 7 level string.
        severity_composite: Phase 4 composite (None if gated).
        severity_band: "low"/"moderate"/"high" or None.
        matched_patterns: list of {name,label,description} dicts (semantic).
        agent_signals: list of {agent_id, signal_strength, direction, quality}
            summarizing what each agent contributed.
        kb_context: grounding snippets from the Allen KB (list of strings).
        audience: "patient" or "clinician" — controls tone/detail.
    """

    user_id: str
    bucket_id: str
    disease: str
    flare_probability: Optional[float]
    confidence_level: str
    severity_composite: Optional[float] = None
    severity_band: Optional[str] = None
    matched_patterns: list = field(default_factory=list)
    agent_signals: list = field(default_factory=list)
    kb_context: list = field(default_factory=list)
    audience: str = "patient"


@dataclass
class TFMResponse:
    """The TFM's explanation output.

    Fields:
        explanation: the natural-language narrative.
        ok: True if produced by the model; False if degraded/fallback.
        model: identifier of the model that produced it (or "fallback").
        error: error string if ok is False.
        prompt_tokens / completion_tokens: usage if the backend reports it.
        trace_id: threaded through for Layer A correlation.
    """

    explanation: str
    ok: bool = True
    model: str = ""
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace_id: str = ""


@runtime_checkable
class ThinkingMachine(Protocol):
    """The swap point. Any model backend implements this one method."""

    name: str

    def explain(self, request: TFMRequest) -> TFMResponse:
        """Produce a grounded explanation of an already-computed evaluation."""
        ...


# --------------------------------------------------------------------------- #
# Shared prompt construction — used by ALL real model backends so that swapping
# Claude <-> local Llama does not change the grounding, guardrails, or framing.
# Only the transport (how the prompt is sent to a model) differs per backend.
# --------------------------------------------------------------------------- #

_SYSTEM_GUARDRAIL = (
    "You are an explanatory assistant inside a health-monitoring system for "
    "people with autoimmune conditions. Your ONLY job is to explain, in clear "
    "and calm language, what the system's already-computed signals mean. "
    "Strict rules: (1) Do NOT give medical advice, diagnoses, treatment, or "
    "dosing. (2) Do NOT invent numbers or certainty the data does not support. "
    "(3) If confidence is low or insufficient, say so plainly and keep the "
    "explanation cautious. (4) Encourage the person to consult their clinician "
    "for medical decisions. (5) Be concise and non-alarming."
)


def build_prompt(request: TFMRequest) -> tuple:
    """Build (system, user) prompt strings from a TFMRequest.

    Centralizing this here is what makes the abstraction real: ClaudeTFM and a
    future LocalLLMTFM send the SAME grounded prompt, so explanation behavior
    is consistent across model backends.
    """
    system = _SYSTEM_GUARDRAIL

    lines = []
    lines.append(f"Condition: {request.disease}")
    lines.append(f"Audience: {request.audience}")
    lines.append(f"Confidence level: {request.confidence_level}")

    if request.flare_probability is None:
        lines.append(
            "Flare probability: NOT AVAILABLE — there was not enough reliable "
            "data this period to estimate it. Explain this honestly and keep "
            "the message reassuring but clear that more data is needed."
        )
    else:
        pct = round(request.flare_probability * 100)
        lines.append(f"Estimated near-term flare likelihood: ~{pct}%")
        if request.severity_band:
            lines.append(f"Overall severity band: {request.severity_band}")

    if request.matched_patterns:
        lines.append("\nObserved corroborating patterns (signals agreeing):")
        for p in request.matched_patterns:
            lines.append(f"  - {p.get('label', p.get('name',''))}: {p.get('description','')}")

    if request.agent_signals:
        lines.append("\nWhat each monitor contributed:")
        for s in request.agent_signals:
            lines.append(
                f"  - {s.get('agent_id','')}: {s.get('direction','')} "
                f"(strength {s.get('signal_strength','?')}, "
                f"data quality {s.get('quality','?')})"
            )

    if request.kb_context:
        lines.append("\nRelevant background (for grounding, do not quote verbatim):")
        for c in request.kb_context:
            lines.append(f"  - {c}")

    lines.append(
        "\nWrite a short explanation (3-5 sentences) for the "
        f"{request.audience}. Explain what these signals collectively suggest, "
        "ground it in the patterns and background, respect the confidence "
        "level, and remind them you are not a substitute for their clinician."
    )

    user = "\n".join(lines)
    return system, user


def fallback_explanation(request: TFMRequest) -> str:
    """Deterministic, safe explanation used when a model is unavailable.

    No model required. Honest and cautious. This is also the basis for MockTFM.
    """
    if request.flare_probability is None:
        return (
            "There wasn't enough reliable data this period to estimate your "
            "flare likelihood. This isn't a cause for alarm on its own — it "
            "usually just means some of your monitors didn't have enough input. "
            "Keeping your logs up to date will improve future estimates. For "
            "any health concerns, please check with your clinician."
        )
    pct = round(request.flare_probability * 100)
    band = request.severity_band or "unclear"
    pattern_note = ""
    if request.matched_patterns:
        labels = ", ".join(
            p.get("label", p.get("name", "")) for p in request.matched_patterns
        )
        pattern_note = f" Several signals are pointing the same direction ({labels})."
    return (
        f"Based on this period's data, your estimated near-term flare likelihood "
        f"is about {pct}% (overall severity: {band}).{pattern_note} This is an "
        f"informational estimate with {request.confidence_level} confidence, not "
        f"a diagnosis. Please consult your clinician for any medical decisions."
    )
