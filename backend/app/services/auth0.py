"""
Auth0 JWT verification service.

Fetches the JWKS from Auth0, caches it in memory, and verifies
incoming Bearer tokens. Returns the decoded claims on success.
"""

import logging
from typing import Optional

import httpx
from jose import jwt, JWTError
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── In-memory JWKS cache ──────────────────────────────────────────────
_jwks_cache: Optional[dict] = None


async def _get_jwks() -> dict:
    """Fetch and cache the Auth0 JSON Web Key Set."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache

    url = f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            logger.info("Auth0: JWKS fetched and cached from %s", url)
            return _jwks_cache
    except Exception as exc:
        logger.error("Auth0: failed to fetch JWKS: %s", exc)
        raise HTTPException(status_code=503, detail="Auth service unavailable")


async def verify_jwt(token: str) -> dict:
    """
    Verify an Auth0 JWT and return the decoded claims.

    Raises HTTPException(401) if the token is invalid or expired.
    """
    if not settings.AUTH0_DOMAIN or not settings.AUTH0_AUDIENCE:
        logger.warning("Auth0: AUTH0_DOMAIN or AUTH0_AUDIENCE not configured — skipping verification")
        return {}

    try:
        jwks = await _get_jwks()
        unverified_header = jwt.get_unverified_header(token)

        # Find the matching key in JWKS
        rsa_key = {}
        for key in jwks.get("keys", []):
            if key.get("kid") == unverified_header.get("kid"):
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n":   key["n"],
                    "e":   key["e"],
                }
                break

        if not rsa_key:
            raise HTTPException(status_code=401, detail="Unable to find matching JWT key")

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=settings.AUTH0_AUDIENCE,
            issuer=f"https://{settings.AUTH0_DOMAIN}/",
        )
        return payload

    except JWTError as exc:
        logger.warning("Auth0: JWT validation failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
