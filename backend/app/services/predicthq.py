"""
PredictHQ API service for the Critic node.
"""

import logging
import httpx
from typing import Optional, List, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

async def get_events(lat: float, lon: float, radius: str = "1mi") -> List[Dict]:
    """ Fetch relevant upcoming events near the coordinates. """
    api_key = settings.PREDICTHQ_API_KEY
    if not api_key:
        logger.warning("PREDICTHQ_API_KEY not set")
        return []

    url = "https://api.predicthq.com/v1/events/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # We ask for events around the specific point
    # rank_level >= 3 generally means moderate to high impact
    params = {
        "within": f"{radius}@{lat},{lon}",
        "limit": 5,
        "sort": "rank"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            events = data.get("results", [])
            return [
                {
                    "title": e.get("title"),
                    "category": e.get("category"),
                    "start": e.get("start"),
                    "rank": e.get("rank")
                }
                for e in events
            ]
    except Exception as exc:
        logger.error("PredictHQ API failed: %s", exc)
        return []
