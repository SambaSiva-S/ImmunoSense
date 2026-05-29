"""Tests for the TFM layer (Challenge 2: swappable abstraction).

Protocol conformance across MockTFM / ClaudeTFM / LocalLLMTFM, mock determinism,
ClaudeTFM fail-safe behavior (no API key -> degraded, never raises), and the
shared grounded prompt construction.
"""

import os

import pytest

from immunosense.tfm import (
    ClaudeTFM,
    LocalLLMTFM,
    MockTFM,
    TFMRequest,
    ThinkingMachine,
    build_prompt,
    fallback_explanation,
)


def _request(prob=0.3, conf="moderate", patterns=None, signals=None,
             kb=None, audience="patient"):
    return TFMRequest(
        patient_id="p1",
        bucket_id="p1_2026-05-27_T2",
        disease="SLE",
        flare_probability=prob,
        confidence_level=conf,
        severity_composite=0.4,
        severity_band="moderate",
        matched_patterns=patterns or [],
        agent_signals=signals or [],
        kb_context=kb or [],
        audience=audience,
    )


class TestProtocolConformance:
    @pytest.mark.parametrize("cls", [MockTFM, ClaudeTFM, LocalLLMTFM])
    def test_all_backends_satisfy_protocol(self, cls):
        tfm = cls() if cls is MockTFM else cls()
        assert isinstance(tfm, ThinkingMachine)
        assert hasattr(tfm, "explain")
        assert tfm.name


class TestMockTFM:
    def test_deterministic(self):
        m = MockTFM()
        req = _request()
        r1 = m.explain(req)
        r2 = m.explain(req)
        assert r1.explanation == r2.explanation
        assert r1.ok is True

    def test_normal_case_has_safety_language(self):
        r = MockTFM().explain(_request(prob=0.3, conf="moderate"))
        text = r.explanation.lower()
        # Must communicate non-prescriptive stance.
        assert "diagnosis" in text or "clinician" in text or "medical" in text

    def test_gated_case_says_not_enough_data(self):
        r = MockTFM().explain(_request(prob=None, conf="insufficient"))
        text = r.explanation.lower()
        assert "enough" in text or "not available" in text or "data" in text

    def test_model_name_recorded(self):
        r = MockTFM().explain(_request())
        assert r.model == "mock-tfm"


class TestClaudeTFMFailSafe:
    def setup_method(self):
        # Ensure no key in env so we exercise the failure path.
        self._saved = os.environ.pop("ANTHROPIC_API_KEY", None)

    def teardown_method(self):
        if self._saved:
            os.environ["ANTHROPIC_API_KEY"] = self._saved

    def test_no_key_degrades_does_not_raise(self):
        c = ClaudeTFM(api_key=None)
        r = c.explain(_request())
        assert r.ok is False
        assert r.model == "fallback"
        assert r.explanation  # still returns safe text
        assert r.error and "key" in r.error.lower()

    def test_fallback_text_is_safe(self):
        c = ClaudeTFM(api_key=None)
        r = c.explain(_request(prob=0.4, conf="moderate"))
        text = r.explanation.lower()
        # Same safety guarantees as MockTFM since fallback is shared.
        assert "diagnosis" not in text or "not a diagnosis" in text

    def test_fail_safe_with_gated_request(self):
        c = ClaudeTFM(api_key=None)
        r = c.explain(_request(prob=None, conf="insufficient"))
        assert r.ok is False
        # Even when gated, fallback respects the gated state.
        assert "enough" in r.explanation.lower() or "data" in r.explanation.lower()


class TestLocalLLMTFMScaffold:
    def test_not_implemented_degrades_safely(self):
        l = LocalLLMTFM()
        r = l.explain(_request())
        assert r.ok is False
        assert "not implemented" in r.error.lower()
        # Still returns a safe fallback so a misconfiguration can't crash anything.
        assert r.explanation


class TestPromptConstruction:
    def test_includes_guardrails(self):
        system, user = build_prompt(_request())
        assert "do not give medical advice" in system.lower() or \
               "do not give" in system.lower()
        assert "diagnos" in system.lower() or "advice" in system.lower()

    def test_includes_grounding_when_provided(self):
        req = _request(
            prob=0.3,
            patterns=[{"name": "autonomic_stress", "label": "Autonomic stress",
                       "description": "HRV + symptoms agree"}],
            signals=[{"agent_id": "agent5_symptoms_mood", "signal_strength": 0.7,
                      "direction": "elevated", "quality": 0.9}],
            kb=["HRV suppression has been associated with flare onset."],
        )
        system, user = build_prompt(req)
        assert "autonomic" in user.lower()
        assert "HRV" in user
        assert "elevated" in user

    def test_gated_prompt_is_honest(self):
        req = _request(prob=None, conf="insufficient")
        system, user = build_prompt(req)
        assert "not available" in user.lower() or "not enough" in user.lower()

    def test_audience_in_prompt(self):
        req = _request(audience="clinician")
        _, user = build_prompt(req)
        assert "clinician" in user.lower()


class TestFallbackExplanation:
    def test_normal_case_includes_probability(self):
        text = fallback_explanation(_request(prob=0.45))
        assert "45%" in text

    def test_gated_case_no_invented_number(self):
        text = fallback_explanation(_request(prob=None, conf="insufficient"))
        # Must NOT invent a probability when gated.
        assert "%" not in text or "data" in text.lower()
        assert "enough" in text.lower() or "not available" in text.lower()
