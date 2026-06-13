"""API endpoint tests — the full Phase 1 surface, dev-auth mode.

Covers: health, auth enforcement, per-agent logging, evaluate (normal + flare),
reports, history, profile/consent, photo URL, the confidence-aware framing, the
tracelog (trace id propagation), and audit logging.
"""

from sqlalchemy import func, select

from server.db.models import AccessLog, BucketReport, SymptomLog


H = {"X-Dev-User": "u_test"}


class TestHealthAndAuth:
    def test_health_open(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_auth_required(self, api_client):
        # No X-Dev-User header -> 401
        assert api_client.post("/v1/evaluate").status_code == 401
        assert api_client.post("/v1/log/symptom", json={"fatigue": 5}).status_code == 401

    def test_trace_id_returned(self, api_client):
        r = api_client.get("/health")
        assert "X-Trace-Id" in r.headers


class TestLogging:
    def test_log_symptom(self, api_client):
        r = api_client.post("/v1/log/symptom", headers=H,
                            json={"source": "tap", "fatigue": 7, "joint_pain": 6})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["bucket_id"].startswith("u_test_")
        assert body["trace_id"]

    def test_log_symptom_persisted(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        with api_client._session_factory() as s:
            n = s.execute(select(func.count()).select_from(SymptomLog)).scalar_one()
        assert n == 1

    def test_log_biomarker(self, api_client):
        r = api_client.post("/v1/log/biomarker", headers=H, json={"crp": 9.2, "esr": 35})
        assert r.status_code == 200

    def test_log_meal(self, api_client):
        r = api_client.post("/v1/log/meal", headers=H,
                            json={"source": "text", "description": "chicken and rice"})
        assert r.status_code == 200


class TestEvaluation:
    def test_evaluate_returns_report(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7, "joint_pain": 6})
        r = api_client.post("/v1/evaluate", headers=H)
        assert r.status_code == 200
        body = r.json()
        assert "confidence_level" in body
        assert "display" in body
        assert "status" in body

    def test_insufficient_gates_number(self, api_client):
        # Single symptom log -> low/insufficient -> no number shown
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 5})
        body = api_client.post("/v1/evaluate", headers=H).json()
        if body["status"] == "insufficient":
            assert body["flare_probability"] is None
            assert body["display"]["show_number"] is False
            assert "keep logging" in body["display"]["headline"].lower() \
                   or "not enough" in body["display"]["headline"].lower()

    def test_flare_button_immediate_report(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 8})
        r = api_client.post("/v1/log/flare", headers=H, json={"severity": 0.9})
        assert r.status_code == 200
        # returns a report directly, not just an ack
        assert "display" in r.json()
        assert "confidence_level" in r.json()

    def test_report_persisted_to_bucket_reports(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        api_client.post("/v1/evaluate", headers=H)
        with api_client._session_factory() as s:
            n = s.execute(select(func.count()).select_from(BucketReport)).scalar_one()
        assert n == 1


class TestReports:
    def test_latest_404_when_empty(self, api_client):
        assert api_client.get("/v1/report/latest", headers=H).status_code == 404

    def test_latest_after_eval(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        api_client.post("/v1/evaluate", headers=H)
        r = api_client.get("/v1/report/latest", headers=H)
        assert r.status_code == 200
        assert r.json()["bucket_id"].startswith("u_test_")

    def test_history(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        api_client.post("/v1/evaluate", headers=H)
        r = api_client.get("/v1/history", headers=H)
        assert r.status_code == 200
        assert len(r.json()["items"]) >= 1


class TestProfileConsent:
    def test_me_empty(self, api_client):
        r = api_client.get("/v1/me", headers=H)
        assert r.status_code == 200
        assert r.json()["user_id"] == "u_test"

    def test_set_and_read_consent(self, api_client):
        api_client.put("/v1/me/consent", headers=H,
                       json={"consent_type": "tfm_ai_processing", "granted": True})
        consents = api_client.get("/v1/me", headers=H).json()["consents"]
        assert consents["tfm_ai_processing"] is True

    def test_revoke_consent(self, api_client):
        api_client.put("/v1/me/consent", headers=H,
                       json={"consent_type": "research_secondary_use", "granted": True})
        api_client.put("/v1/me/consent", headers=H,
                       json={"consent_type": "research_secondary_use", "granted": False})
        consents = api_client.get("/v1/me", headers=H).json()["consents"]
        assert consents["research_secondary_use"] is False


class TestPhoto:
    def test_photo_url_issued(self, api_client):
        r = api_client.post("/v1/photo", headers=H, json={"content_type": "image/jpeg"})
        assert r.status_code == 200
        body = r.json()
        assert body["photo_id"]
        assert body["upload_url"]
        assert body["storage_key"].startswith("u_test/")


class TestAuditAndTrace:
    def test_writes_generate_audit(self, api_client):
        api_client.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        with api_client._session_factory() as s:
            n = s.execute(select(func.count()).select_from(AccessLog)).scalar_one()
        assert n >= 1

    def test_user_isolation(self, api_client):
        # Two users; each sees only their own data.
        api_client.post("/v1/log/symptom", headers={"X-Dev-User": "u_a"}, json={"fatigue": 7})
        api_client.post("/v1/evaluate", headers={"X-Dev-User": "u_a"})
        api_client.post("/v1/log/symptom", headers={"X-Dev-User": "u_b"}, json={"fatigue": 3})
        api_client.post("/v1/evaluate", headers={"X-Dev-User": "u_b"})
        a = api_client.get("/v1/history", headers={"X-Dev-User": "u_a"}).json()["items"]
        b = api_client.get("/v1/history", headers={"X-Dev-User": "u_b"}).json()["items"]
        assert all(i["bucket_id"].startswith("u_a_") for i in a)
        assert all(i["bucket_id"].startswith("u_b_") for i in b)


class TestDebugEndpoint:
    def test_debug_404_when_disabled(self, api_client):
        # api_client fixture uses default settings (debug disabled)
        r = api_client.post("/v1/evaluate/debug", headers={"X-Dev-User": "u_test"})
        assert r.status_code == 404

    def test_debug_returns_agent_detail_when_enabled(self):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from sqlalchemy.orm import sessionmaker
        from fastapi.testclient import TestClient
        from server.db.base import Base
        from server.db import models  # noqa
        from server.api.app import create_app
        from server.api.config import Settings
        e = create_engine("sqlite://", connect_args={"check_same_thread": False},
                          poolclass=StaticPool, future=True)
        Base.metadata.create_all(e)
        sf = sessionmaker(bind=e, expire_on_commit=False, future=True)
        c = TestClient(create_app(session_factory=sf,
                                  settings=Settings(dev_auth=True, enable_debug_endpoint=True)))
        H = {"X-Dev-User": "u_dbg"}
        c.post("/v1/log/symptom", headers=H, json={"fatigue": 7})
        r = c.post("/v1/evaluate/debug", headers=H)
        assert r.status_code == 200
        d = r.json()
        assert "agents" in d and "reporting_agents" in d
        assert "fusion_contributions" in d
        assert any(a["agent_id"] == "agent5_symptoms_mood" for a in d["agents"])


class TestProfile:
    def test_set_and_read_profile(self, api_client):
        H = {"X-Dev-User": "u_prof"}
        r = api_client.put("/v1/me/profile", headers=H,
                           json={"disease": "RA", "timezone": "America/New_York"})
        assert r.status_code == 200
        assert r.json()["disease"] == "RA"
        me = api_client.get("/v1/me", headers=H).json()
        assert me["disease"] == "RA"

    def test_partial_update_keeps_other_field(self, api_client):
        H = {"X-Dev-User": "u_prof2"}
        api_client.put("/v1/me/profile", headers=H, json={"disease": "SLE", "timezone": "UTC"})
        api_client.put("/v1/me/profile", headers=H, json={"timezone": "Asia/Tokyo"})
        me = api_client.get("/v1/me", headers=H).json()
        assert me["disease"] == "SLE"  # unchanged by the tz-only update


class TestBiomarkerBackfill:
    def test_backfill_past_date_lands_in_historical_bucket(self, api_client):
        H = {"X-Dev-User": "u_bf1"}
        r = api_client.post("/v1/log/biomarker", headers=H,
                            json={"crp": 8.2, "esr": 30, "measured_at": "2026-03-15"})
        assert r.status_code == 200
        assert "2026-03-15" in r.json()["bucket_id"]

    def test_future_date_rejected(self, api_client):
        H = {"X-Dev-User": "u_bf2"}
        r = api_client.post("/v1/log/biomarker", headers=H,
                            json={"crp": 5, "measured_at": "2099-01-01"})
        assert r.status_code == 422

    def test_bad_date_format_rejected(self, api_client):
        H = {"X-Dev-User": "u_bf3"}
        r = api_client.post("/v1/log/biomarker", headers=H,
                            json={"crp": 5, "measured_at": "March 3"})
        assert r.status_code == 422

    def test_no_date_defaults_to_now(self, api_client):
        H = {"X-Dev-User": "u_bf4"}
        r = api_client.post("/v1/log/biomarker", headers=H, json={"crp": 6})
        assert r.status_code == 200


class TestPhotoProduction:
    def test_photo_503_when_storage_unconfigured(self):
        # In production mode (dev_auth off) with no Supabase storage config,
        # the endpoint must fail clearly with 503, not return a broken URL.
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from sqlalchemy.orm import sessionmaker
        from fastapi.testclient import TestClient
        from server.db.base import Base
        from server.db import models  # noqa
        from server.api.app import create_app
        from server.api.config import Settings
        e = create_engine("sqlite://", connect_args={"check_same_thread": False},
                          poolclass=StaticPool, future=True)
        Base.metadata.create_all(e)
        sf = sessionmaker(bind=e, expire_on_commit=False, future=True)
        # dev_auth False but no supabase_url/service key -> 503. Use a dev token
        # path by setting dev_auth True for AUTH but simulating prod storage?
        # Simpler: dev_auth must be False to hit the prod branch; supply a fake
        # bearer via X-Dev-User won't work when dev_auth False. So we test the
        # helper-guard directly via a prod-config app with auth bypass off is not
        # feasible here; instead assert the helper raises.
        from server.api.routes import _supabase_signed_upload_url
        import pytest
        from fastapi import HTTPException
        s = Settings(dev_auth=False, supabase_url=None, supabase_service_role_key=None)
        with pytest.raises(HTTPException) as ei:
            _supabase_signed_upload_url(s, "u/2026-06-02/x.jpg")
        assert ei.value.status_code == 503


class TestPhotoView:
    def test_view_own_photo_dev(self, api_client):
        H = {"X-Dev-User": "u_pv"}
        # create a photo via the upload endpoint (dev stub)
        up = api_client.post("/v1/photo", headers=H, json={"content_type": "image/jpeg"}).json()
        pid = up["photo_id"]
        r = api_client.get(f"/v1/photo/{pid}", headers=H)
        assert r.status_code == 200
        assert "view_url" in r.json()

    def test_cannot_view_others_photo(self, api_client):
        A = {"X-Dev-User": "u_pa"}
        B = {"X-Dev-User": "u_pb"}
        up = api_client.post("/v1/photo", headers=A, json={"content_type": "image/jpeg"}).json()
        pid = up["photo_id"]
        # B tries to view A's photo -> 404 (never leaks another user's photo)
        r = api_client.get(f"/v1/photo/{pid}", headers=B)
        assert r.status_code == 404


class TestPhotoList:
    def test_list_photos_returns_own_photos(self, api_client):
        H = {"X-Dev-User": "u_list"}
        # create two photos
        api_client.post("/v1/photo", headers=H, json={"content_type": "image/jpeg"})
        api_client.post("/v1/photo", headers=H, json={"content_type": "image/jpeg"})
        r = api_client.get("/v1/photos", headers=H)
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 2
        assert all("view_url" in i and "uploaded_at" in i for i in items)

    def test_list_photos_is_user_scoped(self, api_client):
        A = {"X-Dev-User": "u_la"}
        B = {"X-Dev-User": "u_lb"}
        api_client.post("/v1/photo", headers=A, json={"content_type": "image/jpeg"})
        # B lists -> should NOT see A's photo
        r = api_client.get("/v1/photos", headers=B)
        assert r.status_code == 200
        assert r.json()["items"] == []
