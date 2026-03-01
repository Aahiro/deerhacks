"""
Gemini API service — thin wrapper for Google Gemini calls.
Uses the GOOGLE_CLOUD_API_KEY for authentication.
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Gemini 1.5 Flash for fast tasks, Pro for multimodal
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


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
        Optional image URLs for multimodal input.

    Returns
    -------
    str | None  — the generated text, or None on failure.
    """
    if not settings.GOOGLE_CLOUD_API_KEY:
        logger.warning("GOOGLE_CLOUD_API_KEY not set — skipping Gemini call")
        return None

    url = f"{_GEMINI_BASE}/{model}:generateContent?key={settings.GOOGLE_CLOUD_API_KEY}"

    # Build parts list
    parts = []

    # Add images if provided (multimodal)
    if image_urls:
        for img_url in image_urls[:3]:  # limit to 3 images to stay frugal
            try:
                async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                    img_resp = await client.get(img_url)
                    img_resp.raise_for_status()
                    import base64
                    img_b64 = base64.b64encode(img_resp.content).decode("utf-8")
                    # Detect content type
                    content_type = img_resp.headers.get("content-type", "image/jpeg")
                    parts.append({
                        "inline_data": {
                            "mime_type": content_type,
                            "data": img_b64,
                        }
                    })
            except Exception as exc:
                logger.warning("Failed to fetch image %s: %s", img_url, exc)

    # Add text prompt
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
