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

    # Auth — when dev_auth is on, an X-Dev-User header stands in for a real JWT.
    # PRODUCTION MUST set DEV_AUTH=0 and provide SUPABASE_JWKS_URL.
    dev_auth: bool = os.environ.get("DEV_AUTH", "1") == "1"
    supabase_jwks_url: str | None = os.environ.get("SUPABASE_JWKS_URL")
    supabase_jwt_audience: str = os.environ.get("SUPABASE_JWT_AUD", "authenticated")

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
