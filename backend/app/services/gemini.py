"""
Gemini API service — thin wrapper for Google Gemini calls.
Uses the GOOGLE_CLOUD_API_KEY for authentication.
"""

import asyncio
import base64
import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Gemini 1.5 Flash for fast tasks, Pro for multimodal
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


async def _fetch_image_part(img_url: str) -> Optional[dict]:
    """Fetch a single image and return a Gemini inline_data part, or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            img_resp = await client.get(img_url)
            img_resp.raise_for_status()
            img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
            content_type = img_resp.headers.get("content-type", "image/jpeg")
            return {"inline_data": {"mime_type": content_type, "data": img_b64}}
    except Exception as exc:
        logger.warning("Failed to fetch image %s: %s", img_url, exc)
        return None


async def generate_content(
    prompt: str,
    model: str = "gemini-2.5-flash",
    image_urls: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Call Gemini and return the text response.

    Parameters
    ----------
    prompt : str
        The text prompt.
    model : str
        Model name (default: gemini-2.5-flash).
    image_urls : list[str] | None
        Optional image URLs for multimodal input. Fetched in parallel (≤3 images).

    Returns
    -------
    str | None  — the generated text, or None on failure.
    """
    if not settings.GOOGLE_CLOUD_API_KEY:
        logger.warning("GOOGLE_CLOUD_API_KEY not set — skipping Gemini call")
        return None

    url = f"{_GEMINI_BASE}/{model}:generateContent?key={settings.GOOGLE_CLOUD_API_KEY}"

    # ── Fetch images in parallel (was sequential — up to 30 s; now ≤8 s) ──
    parts: list[dict] = []
    if image_urls:
        img_results = await asyncio.gather(
            *[_fetch_image_part(u) for u in image_urls[:3]],
            return_exceptions=True,
        )
        parts = [r for r in img_results if isinstance(r, dict)]

    # Add text prompt last
    parts.append({"text": prompt})

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 8192,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            text_parts = content.get("parts", [])
            if text_parts:
                return text_parts[0].get("text", "")
    except httpx.HTTPError as exc:
        logger.error("Gemini request failed: %s", exc)

    return None
