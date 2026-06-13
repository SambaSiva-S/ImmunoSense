"""API configuration, resolved from environment variables.

Phase 1 defaults are dev-friendly (SQLite, dev-auth on) so the whole API runs
locally and in tests without Supabase. Production sets DATABASE_URL,
SUPABASE_JWKS_URL, etc., and turns dev-auth off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Database
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///immunosense_dev.db")
    # Runtime connection for the API. If set, the app connects as the restricted
    # `immunosense_app` role (RLS-enforced) instead of the migration/superuser
    # DATABASE_URL. Migrations always use DATABASE_URL (needs role/DDL privileges);
    # the running app uses this. Falls back to database_url if unset (dev/tests).
    app_database_url: str = os.environ.get("APP_DATABASE_URL", "")

    # Auth — when dev_auth is on, an X-Dev-User header stands in for a real JWT.
    # PRODUCTION MUST set DEV_AUTH=0 and provide SUPABASE_JWKS_URL.
    dev_auth: bool = os.environ.get("DEV_AUTH", "1") == "1"
    supabase_jwks_url: str | None = os.environ.get("SUPABASE_JWKS_URL")
    supabase_jwt_audience: str = os.environ.get("SUPABASE_JWT_AUD", "authenticated")

    # Supabase Storage (meal photos). The service role key is a SECRET — set it
    # via env only, never commit it. Used server-side to mint signed upload URLs
    # against a private bucket. Photos are record-only (no AI reads them).
    supabase_url: str | None = os.environ.get("SUPABASE_URL")
    supabase_service_role_key: str | None = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = os.environ.get("SUPABASE_STORAGE_BUCKET", "meal-photos")

    # Dietary NHANES cache paths (built once via build_dietary_caches.py).
    # If unset, the dietary log endpoint still stores the raw meal, but
    # evaluation will skip dietary rollup (logged as a warning, not an error).
    dietary_density_cache: str | None = os.environ.get("DIETARY_DENSITY_CACHE")
    dietary_food_index_cache: str | None = os.environ.get("DIETARY_FOOD_INDEX_CACHE")

    # TFM — whether to call the live ClaudeTFM. Default off (MockTFM) so the
    # API runs without an API key and tests are deterministic.
    use_claude_tfm: bool = os.environ.get("USE_CLAUDE_TFM", "0") == "1"

    # Disease default for new users without a profile disease set.
    default_disease: str = os.environ.get("DEFAULT_DISEASE", "SLE")

    # --- API hardening ---
    # CORS: comma-separated allowlist of front-end origins. Empty = no browser
    # origin allowed (safe default; the API still works for non-browser clients).
    # Example: CORS_ORIGINS="https://app.immunosense.com,http://localhost:5173"
    cors_origins: str = os.environ.get("CORS_ORIGINS", "")

    # Rate limiting: requests per window per client (IP + user). 0 disables.
    rate_limit_requests: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "120"))
    rate_limit_window_seconds: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
    # Tighter limit for expensive/sensitive endpoints (evaluate, flare, auth).
    rate_limit_heavy_requests: int = int(os.environ.get("RATE_LIMIT_HEAVY_REQUESTS", "20"))

    # Send HSTS header (only meaningful over HTTPS; safe to leave on in prod).
    enable_hsts: bool = os.environ.get("ENABLE_HSTS", "1") == "1"

    # Dev-only agent inspector endpoint (/v1/evaluate/debug). Independent of
    # dev_auth so you can run real JWT auth AND inspect agents while building.
    # MUST be 0 in production — it exposes agent internals.
    enable_debug_endpoint: bool = os.environ.get("ENABLE_DEBUG_ENDPOINT", "0") == "1"

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Test helper — force re-read of env on next get_settings()."""
    global _settings
    _settings = None
