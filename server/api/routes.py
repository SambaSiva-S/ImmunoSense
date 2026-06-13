"""API routes — the Phase 1 endpoint surface.

All log endpoints are per-agent. Evaluation is explicit (/v1/evaluate) plus the
flare button (/v1/log/flare) which evaluates immediately. user_id always comes
from auth, never the body.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select

from server.api.auth import get_current_user_id
from server.api.deps import get_service, get_session_factory, get_settings_dep
from server.api.schemas import (
    BiomarkerLogIn,
    ConsentIn,
    FlareIn,
    LogAck,
    MealLogIn,
    PhotoRequestIn,
    ProfileIn,
    ReportOut,
    SymptomLogIn,
    report_to_out,
)
from server.api.tracelog import get_trace_id, log
from server.db.models import (
    AccessLog,
    BiomarkerReading,
    BucketReport,
    Consent,
    DietaryLog,
    FlareButtonEvent,
    Photo,
    Profile,
    SymptomLog,
)
from immunosense.events import BucketBuilder

router = APIRouter(prefix="/v1")


def _now():
    return datetime.now(timezone.utc)


def _bucket_id(user_id: str, ts: datetime) -> str:
    return BucketBuilder.bucket_for(user_id, ts).bucket_id


def _parse_backfill_date(date_str, fallback):
    """Parse a 'YYYY-MM-DD' backfill date into a UTC datetime anchored at noon
    (noon avoids timezone edges pushing it across a day boundary). Returns
    `fallback` (usually now) when no date is given. Rejects future dates and
    anything implausibly old (>5 years)."""
    if not date_str:
        return fallback
    from fastapi import HTTPException
    try:
        y, m, d = (int(x) for x in str(date_str)[:10].split("-"))
        dt = datetime(y, m, d, 12, 0, 0, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="measured_at must be YYYY-MM-DD")
    now = datetime.now(timezone.utc)
    if dt > now:
        raise HTTPException(status_code=422, detail="measured_at cannot be in the future")
    if (now - dt).days > 365 * 5:
        raise HTTPException(status_code=422, detail="measured_at is too far in the past")
    return dt


def _audit(session, user_id, action, resource):
    session.add(AccessLog(user_id=user_id, accessor_id=user_id,
                          action=action, resource=resource,
                          request_id=get_trace_id()))


# --------------------------------------------------------------------------- #
# Logging endpoints (per-agent)
# --------------------------------------------------------------------------- #
@router.post("/log/symptom", response_model=LogAck)
def log_symptom(body: SymptomLogIn, user_id: str = Depends(get_current_user_id),
                sf=Depends(get_session_factory)):
    ts = _now()
    bucket_id = _bucket_id(user_id, ts)
    log_id = uuid.uuid4()
    with sf() as s:
        s.add(SymptomLog(
            log_id=log_id, user_id=user_id, bucket_id=bucket_id, logged_at=ts,
            source=body.source, fatigue=body.fatigue, joint_pain=body.joint_pain,
            brain_fog_severity=body.brain_fog_severity, gi_distress=body.gi_distress,
            skin_severity=body.skin_severity, sleep_severity=body.sleep_severity,
            energy_severity=body.energy_severity, wellness_severity=body.wellness_severity,
            phq8_score=body.phq8_score, gad7_score=body.gad7_score, free_text=body.free_text,
        ))
        _audit(s, user_id, "write", "health.symptom_logs")
        s.commit()
    return LogAck(log_id=str(log_id), bucket_id=bucket_id, trace_id=get_trace_id())


@router.post("/log/meal", response_model=LogAck)
def log_meal(body: MealLogIn, user_id: str = Depends(get_current_user_id),
             sf=Depends(get_session_factory)):
    ts = _now()
    bucket_id = _bucket_id(user_id, ts)
    log_id = uuid.uuid4()
    with sf() as s:
        s.add(DietaryLog(
            log_id=log_id, user_id=user_id, bucket_id=bucket_id, meal_at=ts,
            source=body.source, description=body.description,
            photo_id=uuid.UUID(body.photo_id) if body.photo_id else None,
        ))
        _audit(s, user_id, "write", "health.dietary_logs")
        s.commit()
    return LogAck(log_id=str(log_id), bucket_id=bucket_id, trace_id=get_trace_id())


@router.post("/log/biomarker", response_model=LogAck)
def log_biomarker(body: BiomarkerLogIn, user_id: str = Depends(get_current_user_id),
                  sf=Depends(get_session_factory)):
    ts = _parse_backfill_date(body.measured_at, _now())
    bucket_id = _bucket_id(user_id, ts)
    reading_id = uuid.uuid4()
    with sf() as s:
        s.add(BiomarkerReading(
            reading_id=reading_id, user_id=user_id, measured_at=ts,
            crp=body.crp, esr=body.esr, payload=body.extra or {},
        ))
        _audit(s, user_id, "write", "health.biomarker_readings")
        s.commit()
    return LogAck(log_id=str(reading_id), bucket_id=bucket_id, trace_id=get_trace_id())


# --------------------------------------------------------------------------- #
# Flare button — logs AND evaluates immediately
# --------------------------------------------------------------------------- #
@router.post("/log/flare", response_model=ReportOut)
def log_flare(body: FlareIn, user_id: str = Depends(get_current_user_id),
              sf=Depends(get_session_factory), svc=Depends(get_service)):
    ts = _now()
    bucket_id = _bucket_id(user_id, ts)
    with sf() as s:
        s.add(FlareButtonEvent(user_id=user_id, pressed_at=ts,
                               severity=body.severity, bucket_id=bucket_id))
        _audit(s, user_id, "write", "health.flare_button_events")
        s.commit()
    # Immediate evaluation (the critical-event override).
    report = svc.evaluate(user_id, ts)
    return report_to_out(report, get_trace_id())


# --------------------------------------------------------------------------- #
# Evaluation (normal path) + reports
# --------------------------------------------------------------------------- #
@router.post("/evaluate", response_model=ReportOut)
def evaluate(user_id: str = Depends(get_current_user_id), svc=Depends(get_service)):
    report = svc.evaluate(user_id, _now())
    return report_to_out(report, get_trace_id())


@router.post("/evaluate/debug")
def evaluate_debug(user_id: str = Depends(get_current_user_id),
                   svc=Depends(get_service), settings=Depends(get_settings_dep)):
    """DEV-ONLY agent inspector. Returns the full agent-level detail the wellness
    response hides. Requires authentication (your own data) AND the
    ENABLE_DEBUG_ENDPOINT flag; returns 404 otherwise so it's never an
    accidental production surface."""
    from fastapi import HTTPException
    if not settings.enable_debug_endpoint:
        raise HTTPException(status_code=404, detail="Not found.")
    report = svc.evaluate(user_id, _now())
    return svc.debug_view(report)


@router.get("/report/latest", response_model=ReportOut)
def report_latest(user_id: str = Depends(get_current_user_id),
                  sf=Depends(get_session_factory)):
    from fastapi import HTTPException
    with sf() as s:
        row = s.execute(
            select(BucketReport).where(BucketReport.user_id == user_id)
            .order_by(BucketReport.evaluated_at.desc())
        ).scalars().first()
        _audit(s, user_id, "read", "health.bucket_reports")
        s.commit()
    if not row:
        raise HTTPException(status_code=404, detail="No reports yet.")
    return _bucketreport_to_out(row)


@router.get("/history")
def history(user_id: str = Depends(get_current_user_id),
            sf=Depends(get_session_factory), limit: int = 30, offset: int = 0):
    with sf() as s:
        rows = s.execute(
            select(BucketReport).where(BucketReport.user_id == user_id)
            .order_by(BucketReport.evaluated_at.desc())
            .limit(min(limit, 100)).offset(offset)
        ).scalars().all()
        _audit(s, user_id, "read", "health.bucket_reports")
        s.commit()
    return {"items": [_bucketreport_to_out(r).model_dump() for r in rows],
            "trace_id": get_trace_id()}


# --------------------------------------------------------------------------- #
# Profile + consent
# --------------------------------------------------------------------------- #
@router.get("/me")
def me(user_id: str = Depends(get_current_user_id), sf=Depends(get_session_factory)):
    with sf() as s:
        prof = s.execute(select(Profile).where(Profile.user_id == user_id)).scalar_one_or_none()
        consents = s.execute(select(Consent).where(Consent.user_id == user_id)).scalars().all()
    return {
        "user_id": user_id,
        "disease": getattr(prof, "disease", None) if prof else None,
        "consents": {c.consent_type: c.granted for c in consents},
        "trace_id": get_trace_id(),
    }


@router.put("/me/profile")
def set_profile(body: ProfileIn, user_id: str = Depends(get_current_user_id),
                sf=Depends(get_session_factory)):
    """Upsert the user's profile (condition + timezone). Used by onboarding."""
    with sf() as s:
        prof = s.execute(select(Profile).where(Profile.user_id == user_id)).scalar_one_or_none()
        if prof is None:
            prof = Profile(user_id=user_id)
            s.add(prof)
        if body.disease is not None:
            prof.disease = body.disease
        if body.timezone is not None:
            prof.timezone = body.timezone
        if body.sex is not None:
            prof.sex = body.sex
        if body.date_of_birth is not None:
            prof.date_of_birth = body.date_of_birth
        if body.height_cm is not None:
            prof.height_cm = body.height_cm
        if body.weight_kg is not None:
            prof.weight_kg = body.weight_kg
        _audit(s, user_id, "write", "identity.profiles")
        s.commit()
        disease, tz = prof.disease, prof.timezone
    return {"ok": True, "disease": disease, "timezone": tz, "trace_id": get_trace_id()}


@router.put("/me/consent")
def set_consent(body: ConsentIn, user_id: str = Depends(get_current_user_id),
                sf=Depends(get_session_factory)):
    with sf() as s:
        existing = s.execute(
            select(Consent).where(Consent.user_id == user_id)
            .where(Consent.consent_type == body.consent_type)
        ).scalar_one_or_none()
        if existing:
            existing.granted = body.granted
            existing.granted_at = _now() if body.granted else existing.granted_at
            existing.revoked_at = None if body.granted else _now()
        else:
            s.add(Consent(user_id=user_id, consent_type=body.consent_type,
                          granted=body.granted,
                          granted_at=_now() if body.granted else None))
        _audit(s, user_id, "write", "identity.consents")
        s.commit()
    return {"ok": True, "consent_type": body.consent_type, "granted": body.granted,
            "trace_id": get_trace_id()}


# --------------------------------------------------------------------------- #
# Photo upload (signed URL) — Supabase Storage on production
# --------------------------------------------------------------------------- #
@router.post("/photo")
def photo_upload_url(body: PhotoRequestIn, user_id: str = Depends(get_current_user_id),
                     sf=Depends(get_session_factory),
                     settings=Depends(get_settings_dep)):
    """Return a signed upload URL + the photo_id to attach to a meal.

    In dev mode (no Supabase Storage), returns a stub URL so the flow is
    exercisable. In production, this issues a real Supabase Storage signed URL.
    """
    photo_id = uuid.uuid4()
    storage_key = f"{user_id}/{datetime.now().strftime('%Y-%m-%d')}/{photo_id}.jpg"
    with sf() as s:
        s.add(Photo(photo_id=photo_id, user_id=user_id, storage_key=storage_key,
                    content_type=body.content_type))
        _audit(s, user_id, "write", "health.photos")
        s.commit()

    if settings.dev_auth:
        upload_url = f"https://dev.local/upload/{storage_key}"  # stub for dev/tests
    else:
        upload_url = _supabase_signed_upload_url(settings, storage_key)

    return {"photo_id": str(photo_id), "upload_url": upload_url,
            "storage_key": storage_key, "trace_id": get_trace_id()}


@router.get("/photo/{photo_id}")
def photo_view_url(photo_id: str, user_id: str = Depends(get_current_user_id),
                   sf=Depends(get_session_factory), settings=Depends(get_settings_dep)):
    """Return a short-lived signed URL to VIEW a photo the user owns.

    Authorization: the photo row must belong to the authenticated user — a user
    can never get a URL for someone else's photo (enforced by the user_id filter
    in the query). Photos remain record-only; this just lets the owner see their
    own image back."""
    from fastapi import HTTPException
    with sf() as s:
        row = s.execute(
            select(Photo).where(Photo.photo_id == uuid.UUID(photo_id),
                                Photo.user_id == user_id)
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Photo not found.")
        _audit(s, user_id, "read", "health.photos")
        s.commit()
        storage_key = row.storage_key

    if settings.dev_auth:
        return {"view_url": f"https://dev.local/view/{storage_key}", "trace_id": get_trace_id()}
    return {"view_url": _supabase_signed_view_url(settings, storage_key),
            "trace_id": get_trace_id()}


def _supabase_signed_view_url(settings, storage_key: str, expires_in: int = 3600) -> str:
    """Mint a short-lived signed DOWNLOAD url for a private-bucket object."""
    from fastapi import HTTPException
    import httpx
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(status_code=503, detail="Photo storage is not configured.")
    base = settings.supabase_url.rstrip("/")
    bucket = settings.supabase_storage_bucket
    endpoint = f"{base}/storage/v1/object/sign/{bucket}/{storage_key}"
    try:
        resp = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {settings.supabase_service_role_key.strip()}",
                     "Content-Type": "application/json"},
            json={"expiresIn": expires_in},
            timeout=10.0,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Storage service unreachable: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Could not sign view URL ({resp.status_code}).")
    signed = resp.json().get("signedURL") or resp.json().get("signedUrl", "")
    return f"{base}/storage/v1{signed}"


def _supabase_signed_upload_url(settings, storage_key: str) -> str:
    """Mint a Supabase Storage signed UPLOAD url for a private bucket.

    Calls the storage REST API with the service role key (server-side secret).
    Returns a fully-qualified URL the client PUTs the file bytes to — the bytes
    never transit our API. Photos are record-only; no AI reads them.
    """
    from fastapi import HTTPException
    import httpx

    if not settings.supabase_url or not settings.supabase_service_role_key:
        # Misconfiguration: fail clearly rather than returning a broken URL.
        raise HTTPException(
            status_code=503,
            detail="Photo storage is not configured (missing SUPABASE_URL or service role key).",
        )
    base = settings.supabase_url.rstrip("/")
    bucket = settings.supabase_storage_bucket
    endpoint = f"{base}/storage/v1/object/upload/sign/{bucket}/{storage_key}"
    try:
        resp = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.supabase_service_role_key.strip()}",
                "Content-Type": "application/json",
            },
            json={},  # Supabase storage rejects an empty body when content-type is JSON
            timeout=10.0,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Storage service unreachable: {e}")
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502,
                            detail=f"Could not create upload URL ({resp.status_code}).")
    # Supabase returns {"url": "/object/upload/sign/<bucket>/<key>?token=..."}.
    token_path = resp.json().get("url", "")
    return f"{base}/storage/v1{token_path}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _bucketreport_to_out(row) -> ReportOut:
    from server.api.schemas import DisplayBlock, PatternOut, build_display
    display = build_display(row.confidence_level, row.flare_probability, row.severity_band)
    patterns = [PatternOut(name=p.get("name", ""), label=p.get("label", ""),
                           description=p.get("description", ""))
                for p in (row.matched_patterns or [])] if isinstance(row.matched_patterns, list) else []
    return ReportOut(
        bucket_id=row.bucket_id,
        evaluated_at=row.evaluated_at.isoformat() if row.evaluated_at else None,
        confidence_level=row.confidence_level,
        status="insufficient" if row.flare_probability is None else "ok",
        flare_probability=row.flare_probability,
        severity_composite=row.severity_composite,
        severity_band=row.severity_band,
        matched_patterns=patterns,
        explanation=row.explanation,
        display=display,
        trace_id=get_trace_id(),
    )
