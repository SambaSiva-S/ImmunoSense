"""Verify the API layer end-to-end (dev-auth, in-memory DB).

Module-level imports. Drives the full loop: log -> evaluate -> report, plus
auth enforcement, the confidence-aware framing, tracelog, and audit logging.

Run from project root:
    venv\\Scripts\\python.exe verify_api.py
"""

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from server.db.base import Base
from server.db import models
from server.db.models import AccessLog, BucketReport
from server.api.app import create_app
from server.api.config import Settings


def main():
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("API verification")
    print("=" * 50)

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    app = create_app(session_factory=sf, settings=Settings(dev_auth=True))
    client = TestClient(app)
    H = {"X-Dev-User": "u_verify"}

    print("\n1. Health + auth")
    check("health open", client.get("/health").json()["status"] == "ok")
    check("trace id header present", "X-Trace-Id" in client.get("/health").headers)
    check("no-auth evaluate -> 401", client.post("/v1/evaluate").status_code == 401)

    print("\n2. Per-agent logging")
    check("log symptom 200",
          client.post("/v1/log/symptom", headers=H,
                      json={"source": "tap", "fatigue": 7, "joint_pain": 6}).status_code == 200)
    check("log second symptom 200",
          client.post("/v1/log/symptom", headers=H,
                      json={"source": "voice", "fatigue": 8, "free_text": "rough"}).status_code == 200)
    check("log biomarker 200",
          client.post("/v1/log/biomarker", headers=H,
                      json={"crp": 9.2, "esr": 35}).status_code == 200)
    check("log meal 200",
          client.post("/v1/log/meal", headers=H,
                      json={"source": "text", "description": "chicken and rice"}).status_code == 200)

    print("\n3. Evaluate (normal path)")
    r = client.post("/v1/evaluate", headers=H)
    check("evaluate 200", r.status_code == 200)
    body = r.json()
    check("response has confidence_level", "confidence_level" in body)
    check("response has display block", "display" in body)
    check("confidence-aware: gated number when insufficient",
          (body["status"] != "insufficient") or (body["flare_probability"] is None
           and body["display"]["show_number"] is False))

    print("\n4. Flare button (immediate eval)")
    r = client.post("/v1/log/flare", headers=H, json={"severity": 0.9})
    check("flare returns a report", r.status_code == 200 and "display" in r.json())

    print("\n5. Reports + history")
    check("report/latest 200", client.get("/v1/report/latest", headers=H).status_code == 200)
    check("history has items", len(client.get("/v1/history", headers=H).json()["items"]) >= 1)
    with sf() as s:
        n_reports = s.execute(select(func.count()).select_from(BucketReport)).scalar_one()
    check("bucket_reports persisted", n_reports >= 1)

    print("\n6. Profile + consent")
    client.put("/v1/me/consent", headers=H,
               json={"consent_type": "tfm_ai_processing", "granted": True})
    check("consent set + read",
          client.get("/v1/me", headers=H).json()["consents"].get("tfm_ai_processing") is True)

    print("\n7. Photo upload URL")
    r = client.post("/v1/photo", headers=H, json={"content_type": "image/jpeg"})
    check("photo url issued", r.status_code == 200 and bool(r.json().get("photo_id")))

    print("\n8. Audit + isolation")
    with sf() as s:
        n_audit = s.execute(select(func.count()).select_from(AccessLog)).scalar_one()
    check("audit rows written", n_audit > 0)
    client.post("/v1/log/symptom", headers={"X-Dev-User": "u_other"}, json={"fatigue": 2})
    client.post("/v1/evaluate", headers={"X-Dev-User": "u_other"})
    mine = client.get("/v1/history", headers=H).json()["items"]
    check("user isolation (only own buckets)",
          all(i["bucket_id"].startswith("u_verify_") for i in mine))

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"RESULT: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("API verified OK (dev-auth mode).")
        print("\nNote: real Supabase JWT auth + photo storage need your Supabase")
        print("project — see server/api/README.md for the wiring + verification steps.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
