"""Authentication — resolve the request's user_id.

Two modes:
  - PRODUCTION (dev_auth=False): validate the Supabase JWT from the
    Authorization: Bearer header against Supabase's JWKS, extract the `sub`
    claim (the Supabase auth user id) -> that IS the user_id.
  - DEV/TEST (dev_auth=True): an `X-Dev-User` header supplies the user_id
    directly, so the whole API is runnable locally and in CI without Supabase.

The user_id is ALWAYS derived from the token (or dev header), NEVER trusted from
a request body. This is the security boundary.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from server.api.config import Settings, get_settings


class AuthError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


@lru_cache(maxsize=1)
def _jwks_client(jwks_url: str):
    # Imported lazily so the module loads without PyJWT's crypto extras in dev.
    import jwt  # PyJWT
    return jwt.PyJWKClient(jwks_url)


def _verify_supabase_jwt(token: str, settings: Settings) -> str:
    import jwt  # PyJWT
    if not settings.supabase_jwks_url:
        raise AuthError("Auth not configured (SUPABASE_JWKS_URL missing).")
    try:
        signing_key = _jwks_client(settings.supabase_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.supabase_jwt_audience,
        )
    except Exception as exc:  # noqa: BLE001
        raise AuthError(f"Invalid token: {type(exc).__name__}")
    sub = claims.get("sub")
    if not sub:
        raise AuthError("Token missing 'sub' claim.")
    return str(sub)


async def get_current_user_id(
    authorization: str | None = Header(default=None),
    x_dev_user: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency: returns the authenticated user_id."""
    if settings.dev_auth:
        if not x_dev_user:
            raise AuthError("Dev auth on: supply X-Dev-User header.")
        return x_dev_user

    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    return _verify_supabase_jwt(token, settings)
