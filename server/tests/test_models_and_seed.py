"""Tests for the data model: schema creation, PHI marking, the encryption
seam (Phase 1 passthrough), and the reference-data seeder."""

import uuid
from datetime import datetime, timezone

import pytest

from server.db.models import (
    BiomarkerReading,
    CalibrationLR,
    Consent,
    CorroborationPatternRow,
    DietaryLog,
    Profile,
    SymptomLog,
    User,
)
from server.db.seed import seed_reference_data
from server.db.types import EncryptedString


class TestSchemaCreation:
    def test_all_tables_exist(self, session_factory):
        from sqlalchemy import inspect
        engine = session_factory.kw["bind"]
        tables = set(inspect(engine).get_table_names())
        # Spot-check one table per schema area.
        for t in ("users", "events", "symptom_logs", "calibration_lr",
                  "access_log", "photos", "consents"):
            assert t in tables, f"missing table {t}"


class TestIdentitySeparation:
    def test_user_and_profile_keyed_by_user_id(self, session_factory):
        uid = uuid.uuid4()
        with session_factory() as s:
            s.add(User(user_id=uid, email="a@example.com"))
            s.add(Profile(user_id=uid, display_name="Test", disease="SLE"))
            s.commit()
        with session_factory() as s:
            u = s.get(User, uid)
            assert u.email == "a@example.com"
            p = s.get(Profile, uid)
            assert p.disease == "SLE"

    def test_email_only_lives_on_user(self):
        # Structural guarantee: no operational health table has an email column.
        from server.db import models
        for model_name in ("SymptomLog", "DietaryLog", "BiomarkerReading",
                            "EventRow", "BucketReport", "Photo"):
            model = getattr(models, model_name)
            cols = set(model.__table__.columns.keys())
            assert "email" not in cols, f"{model_name} must not carry email"
            assert "user_id" in cols, f"{model_name} must key on user_id"


class TestPHIMarking:
    def test_phi_columns_use_encrypted_type(self):
        # The free_text / description / explanation / email columns must be PHI-marked.
        assert isinstance(SymptomLog.__table__.c.free_text.type, EncryptedString)
        assert isinstance(DietaryLog.__table__.c.description.type, EncryptedString)
        assert isinstance(User.__table__.c.email.type, EncryptedString)

    def test_encryption_seam_passthrough_in_phase1(self, session_factory, monkeypatch):
        # With no IMMUNOSENSE_PHI_KEY, EncryptedString stores plaintext (no-op).
        monkeypatch.delenv("IMMUNOSENSE_PHI_KEY", raising=False)
        with session_factory() as s:
            log = SymptomLog(user_id="u1", bucket_id="u1_2026-05-27_T2",
                             logged_at=datetime.now(timezone.utc),
                             fatigue=7.0, free_text="felt rough today")
            s.add(log)
            s.commit()
            lid = log.log_id
        with session_factory() as s:
            got = s.get(SymptomLog, lid)
            assert got.free_text == "felt rough today"  # round-trips as plaintext
            assert got.fatigue == 7.0

    def test_phi_columns_discoverable(self):
        # Tooling can enumerate PHI columns via the is_phi marker.
        phi_cols = [
            c.name for c in SymptomLog.__table__.columns
            if getattr(c.type, "is_phi", False)
        ]
        assert "free_text" in phi_cols


class TestSeeder:
    def test_seed_reference_data(self, session_factory):
        summary = seed_reference_data(session_factory)
        assert summary["calibration_version"] == "lr-v1"
        assert summary["lr_rows"] == 5      # 5 agents
        assert 6 <= summary["pattern_rows"] <= 8  # 6-8 patterns

        with session_factory() as s:
            from sqlalchemy import select, func
            n_lr = s.execute(select(func.count()).select_from(CalibrationLR)).scalar_one()
            n_pat = s.execute(select(func.count()).select_from(CorroborationPatternRow)).scalar_one()
        assert n_lr == 5
        assert 6 <= n_pat <= 8

    def test_seed_is_idempotent(self, session_factory):
        seed_reference_data(session_factory)
        seed_reference_data(session_factory)  # run twice
        with session_factory() as s:
            from sqlalchemy import select, func
            n_lr = s.execute(select(func.count()).select_from(CalibrationLR)).scalar_one()
        assert n_lr == 5  # not duplicated

    def test_seeded_lr_has_provenance(self, session_factory):
        seed_reference_data(session_factory)
        with session_factory() as s:
            from sqlalchemy import select
            rows = s.execute(select(CalibrationLR)).scalars().all()
        assert all(r.source for r in rows)


class TestConsent:
    def test_consent_record(self, session_factory):
        uid = uuid.uuid4()
        with session_factory() as s:
            s.add(Consent(user_id=uid, consent_type="tfm_ai_processing",
                          granted=True, granted_at=datetime.now(timezone.utc)))
            s.commit()
        with session_factory() as s:
            from sqlalchemy import select
            c = s.execute(select(Consent).where(Consent.user_id == uid)).scalar_one()
            assert c.consent_type == "tfm_ai_processing"
            assert c.granted is True
