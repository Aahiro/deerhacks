import logging
import httpx
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class Auth0Service:
    def __init__(self):
        self.domain = settings.AUTH0_DOMAIN
        self.client_id = settings.AUTH0_CLIENT_ID
        self.client_secret = settings.AUTH0_CLIENT_SECRET
        self.audience = settings.AUTH0_AUDIENCE
        
    async def get_management_token(self) -> Optional[str]:
        """Fetch a Machine-to-Machine token for the Auth0 Management API."""
        if not all([self.domain, self.client_id, self.client_secret]):
            logger.warning("Auth0 configuration missing. Cannot fetch management token.")
            return None
            
        url = f"https://{self.domain}/oauth/token"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": f"https://{self.domain}/api/v2/",
            "grant_type": "client_credentials"
        }
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("access_token")
        except Exception as e:
            logger.error(f"Failed to get Auth0 Management token: {e}")
            return None

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user metadata from Auth0 to personalize agent weights."""
        if not user_id:
            return {}
            
        token = await self.get_management_token()
        if not token:
            return {}
            
        url = f"https://{self.domain}/api/v2/users/{user_id}"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                user_data = resp.json()
                
                # We specifically care about app_metadata and user_metadata
                # which would hold profile info like "student", "senior", "budget", etc.
                profile = {}
                profile.update(user_data.get("user_metadata", {}))
                profile.update(user_data.get("app_metadata", {}))
                
                return profile
        except Exception as e:
            logger.error(f"Failed to fetch user profile for {user_id}: {e}")
            return {}

    async def get_idp_token(self, user_id: str, provider: str = "google-oauth2") -> Optional[str]:
        """Retrieve a third-party Identity Provider token (e.g., Google Calendar) via Token Vault."""
        if not user_id:
            return None
            
        token = await self.get_management_token()
        if not token:
            return None
            
        url = f"https://{self.domain}/api/v2/users/{user_id}/identities"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                identities = resp.json()
                
                # Find the requested identity provider
                for ext_id in identities:
                    if ext_id.get("provider") == provider:
                        return ext_id.get("access_token")
                        
                logger.warning(f"No active {provider} identity found for {user_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch {provider} token from Auth0 Vault: {e}")
            return None

    async def trigger_ciba_auth(self, user_id: str, message: str) -> Optional[str]:
        """
        Trigger an Asynchronous Authorization (CIBA) push notification to the user's device.
        Returns an 'auth_req_id' string if successful, which must be polled later.
        """
        if not all([self.domain, self.client_id, self.client_secret]):
            return None
            
        url = f"https://{self.domain}/oauth/bc-authorize"
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "login_hint": user_id, 
            "binding_message": message
        }
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, data=payload)
                resp.raise_for_status()
                return resp.json().get("auth_req_id")
        except Exception as e:
            logger.error(f"Failed to trigger CIBA for {user_id}: {e}")
            return None

    async def poll_ciba_status(self, auth_req_id: str) -> Dict[str, Any]:
        """
        Poll the status of a pending CIBA authorization.
        Returns a dictionary with status: "pending", "approved", or "rejected".
        """
        if not all([self.domain, self.client_id, self.client_secret]):
            return {"status": "error", "detail": "Missing Auth0 Config"}
            
        url = f"https://{self.domain}/oauth/token"
        payload = {
            "grant_type": "urn:openid:params:grant-type:ciba",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "auth_req_id": auth_req_id
        }
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, data=payload)
                
                if resp.status_code == 200:
                    # Token granted = Approved by user
                    return {"status": "approved", "access_token": resp.json().get("access_token")}
                    
                # 400 Bad Request indicates it's still pending, rejected, or expired
                err_data = resp.json()
                error_type = err_data.get("error")
                
                if error_type == "authorization_pending":
                    return {"status": "pending"}
                elif error_type == "access_denied":
                    return {"status": "rejected", "detail": "User denied the request"}
                elif error_type == "expired_token":
                    return {"status": "error", "detail": "CIBA request expired"}
                else:
                    return {"status": "error", "detail": error_type}
                    
        except Exception as e:
            logger.error(f"CIBA polling failed: {e}")
            return {"status": "error", "detail": str(e)}

# Export an instance
auth0_service = Auth0Service()


# ── JWT Verification (for API routes) ────────────────────────────────────────
# Simple JWKS-based JWT verification for incoming Bearer tokens.
# Caches the JWKS to avoid a network round-trip on every request.

_jwks_cache: dict = {}

async def verify_jwt(token: str) -> dict:
    """
    Decode and verify an Auth0 JWT Bearer token.

    Fetches JWKS from Auth0 on first call, then caches it.
    Raises HTTPException(401) for invalid / expired tokens.
    Returns the decoded payload dict on success.
    """
    from fastapi import HTTPException

    domain = settings.AUTH0_DOMAIN
    audience = settings.AUTH0_AUDIENCE

    if not domain or not audience:
        # Auth0 not configured — treat every token as invalid
        raise HTTPException(status_code=401, detail="Auth0 not configured")

    global _jwks_cache
    if not _jwks_cache:
        jwks_url = f"https://{domain}/.well-known/jwks.json"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(jwks_url)
                resp.raise_for_status()
                _jwks_cache = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch JWKS: %s", exc)
            raise HTTPException(status_code=401, detail="Could not fetch JWKS")

    try:
        from jose import jwt as jose_jwt, JWTError
        payload = jose_jwt.decode(
            token,
            _jwks_cache,
            algorithms=["RS256"],
            audience=audience,
            issuer=f"https://{domain}/",
        )
        return payload
    except Exception as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
