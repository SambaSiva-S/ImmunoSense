"""SQLAlchemy models for the four schemas: identity, health, reference, audit.

PHI columns use EncryptedString (the encryption seam — passthrough in Phase 1).
Everything operational keys on user_id (the locked identifier). Email lives ONLY
in identity.users. This mirrors the data model in ARCHITECTURE_DECISIONS.md.

Wearable and environment tables exist but are "dormant" in Phase 1 — the schema
supports them; live data wiring waits on device integration (wearable) and a
provider + location (environment).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import (
    SCHEMA_AUDIT,
    SCHEMA_HEALTH,
    SCHEMA_IDENTITY,
    SCHEMA_REFERENCE,
    Base,
    schema_for,
)
from server.db.types import GUID, EncryptedString, JSONType


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ===================================================================== #
# identity schema
# ===================================================================== #
class User(Base):
    """The account. On Supabase this links to auth.users; here we model the
    minimal fields the app owns. Email is the ONLY PHI in this schema."""

    __tablename__ = "users"
    __table_args__ = {"schema": schema_for(SCHEMA_IDENTITY)}

    user_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(EncryptedString(320), unique=True)  # PHI
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Profile(Base):
    __tablename__ = "profiles"
    __table_args__ = {"schema": schema_for(SCHEMA_IDENTITY)}

    user_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    disease: Mapped[str | None] = mapped_column(String(64), nullable=True)  # "SLE", "RA"
    date_of_birth: Mapped[str | None] = mapped_column(EncryptedString(32), nullable=True)  # PHI
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Consent(Base):
    """One row per consent decision. Enforcement reads this (D4/D5)."""

    __tablename__ = "consents"
    __table_args__ = {"schema": schema_for(SCHEMA_IDENTITY)}

    consent_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(GUID, index=True)
    consent_type: Mapped[str] = mapped_column(String(64))  # "tfm_ai_processing", "research_secondary_use"
    granted: Mapped[bool] = mapped_column(Boolean, default=False)
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ===================================================================== #
# health schema (all PHI — every row links to user_id)
# ===================================================================== #
class EventRow(Base):
    """Layer A event, Postgres-backed (replaces NDJSON files via PostgresEventLog)."""

    __tablename__ = "events"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)  # PHI link
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    bucket_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONType)  # PHI content
    quality: Mapped[float] = mapped_column(Float, default=0.0)
    trace_id: Mapped[str] = mapped_column(String(64), default="")
    schema_version: Mapped[str] = mapped_column(String(16), default="v1")


class BucketReport(Base):
    """One ConductorReport summary per (user, bucket)."""

    __tablename__ = "bucket_reports"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    bucket_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    confidence_level: Mapped[str] = mapped_column(String(16))
    overall_quality: Mapped[float] = mapped_column(Float, default=0.0)
    flare_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity_composite: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity_band: Mapped[str | None] = mapped_column(String(16), nullable=True)
    matched_patterns: Mapped[dict] = mapped_column(JSONType, default=list)
    decision: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # PHI
    explanation: Mapped[str | None] = mapped_column(EncryptedString(4000), nullable=True)  # PHI
    calibration_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), default="")


class SymptomLog(Base):
    __tablename__ = "symptom_logs"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    log_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    bucket_id: Mapped[str] = mapped_column(String(128), index=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[str] = mapped_column(String(24), default="tap")  # tap|text|voice
    # the symptom fields agent5 consumes (PHI)
    fatigue: Mapped[float | None] = mapped_column(Float, nullable=True)
    joint_pain: Mapped[float | None] = mapped_column(Float, nullable=True)
    brain_fog_severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    gi_distress: Mapped[float | None] = mapped_column(Float, nullable=True)
    skin_severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    wellness_severity: Mapped[float | None] = mapped_column(Float, nullable=True)
    phq8_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    gad7_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    free_text: Mapped[str | None] = mapped_column(EncryptedString(2000), nullable=True)  # PHI


class DietaryLog(Base):
    __tablename__ = "dietary_logs"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    log_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    bucket_id: Mapped[str] = mapped_column(String(128), index=True)
    meal_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[str] = mapped_column(String(24), default="tap")  # tap|text|voice|photo
    description: Mapped[str | None] = mapped_column(EncryptedString(2000), nullable=True)  # PHI
    photo_id: Mapped[uuid.UUID | None] = mapped_column(GUID, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)  # structured food data


class BiomarkerReading(Base):
    __tablename__ = "biomarker_readings"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    reading_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # self-entered lab values (PHI)
    crp: Mapped[float | None] = mapped_column(Float, nullable=True)
    esr: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)  # other biomarker fields


class WearableReading(Base):
    """Dormant in Phase 1 (needs device integration). Schema ready."""

    __tablename__ = "wearable_readings"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    reading_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    night_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    hrv_rmssd: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)


class Photo(Base):
    """Food photo metadata. The file lives in object storage; this is the row."""

    __tablename__ = "photos"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    photo_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    bucket_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(512))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    content_type: Mapped[str] = mapped_column(String(64), default="image/jpeg")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    user_description: Mapped[str | None] = mapped_column(EncryptedString(2000), nullable=True)  # PHI
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class FlareButtonEvent(Base):
    __tablename__ = "flare_button_events"
    __table_args__ = {"schema": schema_for(SCHEMA_HEALTH)}

    event_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    pressed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    severity: Mapped[float] = mapped_column(Float)
    bucket_id: Mapped[str] = mapped_column(String(128))


# ===================================================================== #
# reference schema (read-only population data, no PHI)
# ===================================================================== #
class CalibrationLR(Base):
    __tablename__ = "calibration_lr"
    __table_args__ = {"schema": schema_for(SCHEMA_REFERENCE)}

    version: Mapped[str] = mapped_column(String(16), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    lr_positive: Mapped[float] = mapped_column(Float)
    lr_negative: Mapped[float] = mapped_column(Float)
    signal_threshold: Mapped[float] = mapped_column(Float)
    low_threshold: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(Text)


class CorroborationPatternRow(Base):
    __tablename__ = "corroboration_patterns"
    __table_args__ = {"schema": schema_for(SCHEMA_REFERENCE)}

    pattern_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    required_agents: Mapped[dict] = mapped_column(JSONType)
    optional_agents: Mapped[dict] = mapped_column(JSONType, default=list)
    source: Mapped[str] = mapped_column(Text)


class KnowledgeEntryRow(Base):
    """Starts empty (NullKB). Real KB content loaded later."""

    __tablename__ = "knowledge_entries"
    __table_args__ = {"schema": schema_for(SCHEMA_REFERENCE)}

    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    diseases: Mapped[dict] = mapped_column(JSONType, default=list)
    tags: Mapped[dict] = mapped_column(JSONType, default=list)
    source: Mapped[str] = mapped_column(Text)


# ===================================================================== #
# audit schema
# ===================================================================== #
class AccessLog(Base):
    __tablename__ = "access_log"
    __table_args__ = {"schema": schema_for(SCHEMA_AUDIT)}

    log_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    accessor_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action: Mapped[str] = mapped_column(String(16))  # read|write|delete
    resource: Mapped[str] = mapped_column(String(128))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DeleteLog(Base):
    __tablename__ = "delete_log"
    __table_args__ = {"schema": schema_for(SCHEMA_AUDIT)}

    delete_id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    resource: Mapped[str] = mapped_column(String(128))
    resource_id: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
