"""
Yelp Fusion API service — supplementary venue discovery.
All API keys stay server-side; nothing is exposed to the frontend.
"""

import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"


async def search_yelp(
    term: str,
    location: str = "Toronto, ON",
    max_results: int = 10,
) -> list[dict]:
    """
    Search Yelp Fusion and return normalised venue dicts.

    Parameters
    ----------
    term : str
        Search term (e.g. "basketball courts").
    location : str
        Location string (e.g. "downtown Toronto").
    max_results : int
        Cap on returned results.

    Returns
    -------
    list[dict]  — same shape as Google Places output for easy merging.
    """
    if not settings.YELP_API_KEY:
        logger.warning("YELP_API_KEY not set — skipping Yelp")
        return []

    headers = {"Authorization": f"Bearer {settings.YELP_API_KEY}"}

    params = {
        "term": term,
        "location": location,
        "limit": min(max_results, 50),
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(_SEARCH_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.error("Yelp request failed: %s", exc)
        return []

    results = []
    for biz in data.get("businesses", []):
        coords = biz.get("coordinates", {})
        results.append({
            "venue_id": f"yelp_{biz.get('id', '')}",
            "name": biz.get("name", "Unknown"),
            "address": ", ".join(biz.get("location", {}).get("display_address", [])),
            "lat": float(coords.get("latitude", 0)),
            "lng": float(coords.get("longitude", 0)),
            "rating": float(biz.get("rating", 0)),
            "review_count": int(biz.get("review_count", 0)),
            "photos": [biz.get("image_url", "")] if biz.get("image_url") else [],
            "category": biz.get("categories", [{}])[0].get("alias", "") if biz.get("categories") else "",
            "price_range": biz.get("price"),
            "website": biz.get("url", ""),
            "source": "yelp",
        })

    return results[:max_results]
