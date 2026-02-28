"""
FastAPI dependencies for authentication.

Uses HTTPBearer with auto_error=False so that unauthenticated
requests are allowed through — the pipeline degrades gracefully
rather than hard-blocking users without a token.
"""

from typing import Optional

from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.auth0 import verify_jwt

_bearer = HTTPBearer(auto_error=False)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    Extract and verify the Auth0 Bearer token if present.

    Returns the decoded JWT claims (including app_metadata) when a valid
    token is provided. Returns an empty dict when no token is present,
    allowing unauthenticated requests to proceed normally.
    """
    if credentials is None:
        return {}
    return await verify_jwt(credentials.credentials)
