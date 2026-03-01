"""
ElevenLabs Text-to-Speech service wrapper.
"""

import logging
import httpx
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default voice IDs for different agent personalities
VOICE_PRESETS = {
    "default": "21m00Tcm4TlvDq8ikWAM",      # Rachel — neutral, clear
    "commander": "21m00Tcm4TlvDq8ikWAM",     # Rachel — authoritative
    "critic": "VR6AewLTigWG4xSOukaG",        # Arnold — skeptical tone
    "vibe_matcher": "EXAVITQu4vr4xnSDxMaL",  # Bella — enthusiastic
    "cost_analyst": "ErXwobaYiN019PkySvjV",   # Antoni — measured, precise
}


async def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    model_id: str = "eleven_monolingual_v1",
) -> Optional[bytes]:
    """
    Convert text to speech using ElevenLabs API.

    Args:
        text: The text to synthesize.
        voice_id: ElevenLabs voice ID (defaults to Rachel).
        model_id: ElevenLabs model to use.

    Returns:
        Audio bytes (MP3) or None if the API call fails.
    """
    api_key = settings.ELEVENLABS_API_KEY
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not set — skipping TTS.")
        return None

    if not voice_id:
        voice_id = VOICE_PRESETS["default"]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content
    except httpx.HTTPStatusError as e:
        logger.error(f"ElevenLabs API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {e}")
        return None
